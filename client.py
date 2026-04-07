"""Incident Diagnosis Environment Client.

Typed wrapper around ``EnvClient`` for the IncidentDiagnosisEnv.

Typical usage (sync):
    >>> client = IncidentDiagnosisEnv(base_url="http://localhost:8000")
    >>> with client.sync() as env:
    ...     result = env.reset(task_id="easy")
    ...     obs = result.observation
    ...     action = IncidentDiagnosisAction(
    ...         action_type="inspect_service", target="database"
    ...     )
    ...     result = env.step(action)
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import (
    IncidentDiagnosisAction,
    IncidentDiagnosisObservation,
    ServiceStatus,
)


class IncidentDiagnosisEnv(
    EnvClient[IncidentDiagnosisAction, IncidentDiagnosisObservation, State]
):
    """Typed client for the Incident Diagnosis environment."""

    def _step_payload(self, action: IncidentDiagnosisAction) -> Dict[str, Any]:
        """Serialise action to JSON, omitting None fields."""
        payload: Dict[str, Any] = {"action_type": action.action_type}
        for field in ("target", "query", "patch_payload", "diagnosis"):
            value = getattr(action, field, None)
            if value is not None:
                payload[field] = value
        return payload

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[IncidentDiagnosisObservation]:
        obs_data = payload.get("observation", {})

        # Reconstruct ServiceStatus list
        raw_statuses = obs_data.get("service_statuses", [])
        service_statuses = [
            ServiceStatus(
                name       = s["name"],
                status     = s["status"],
                cpu_pct    = s.get("cpu_pct"),
                mem_pct    = s.get("mem_pct"),
                error_rate = s.get("error_rate"),
            )
            for s in raw_statuses
            if isinstance(s, dict)
        ]

        observation = IncidentDiagnosisObservation(
            visible_logs      = obs_data.get("visible_logs", []),
            service_statuses  = service_statuses,
            dependency_health = obs_data.get("dependency_health", {}),
            alerts            = obs_data.get("alerts", []),
            previous_actions  = obs_data.get("previous_actions", []),
            step_count        = int(obs_data.get("step_count", 0)),
            done              = bool(payload.get("done", obs_data.get("done", False))),
            metadata          = obs_data.get("metadata", {}),
        )

        metadata = obs_data.get("metadata", {})
        # Prefer top-level payload reward, then obs.reward field, then metadata fallback
        raw_reward = payload.get("reward")
        if raw_reward is None:
            raw_reward = obs_data.get("reward")   # dedicated field on the observation
        if raw_reward is None:
            raw_reward = metadata.get("reward")

        return StepResult(
            observation = observation,
            reward      = raw_reward,
            done        = bool(payload.get("done", obs_data.get("done", False))),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id = payload.get("episode_id"),
            step_count = int(payload.get("step_count", 0)),
        )
