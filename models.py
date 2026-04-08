"""
Data models for the Incident Diagnosis Environment.

Defines models in dependency order:
  1. ServiceStatus         — status snapshot of a single service node
  2. IncidentDiagnosisAction  — action an agent submits each step
  3. IncidentDiagnosisObservation — partial observation returned after reset()/step()

Design notes
------------
* The agent NEVER sees the true hidden SystemState.
* Observation is deliberately partial / noisy (POMDP).
* Actions carry an optional `patch_payload` string for apply_patch actions.
"""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field, field_validator


# 1. ServiceStatus (embedded in Observation)

class ServiceStatus(BaseModel):
    """
    A lightweight status snapshot of a single service, as visible to the agent.

    Values can be noisy / incomplete — the environment intentionally adds noise
    to prevent simple pattern-matching.
    """

    name: str = Field(..., description="Service name (e.g. 'auth_service').")
    status: Literal["healthy", "degraded", "down", "unknown"] = Field(
        ..., description="Observed status of the service."
    )
    cpu_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="CPU utilisation percentage visible to the agent, or None if not yet inspected.",
    )
    mem_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Memory utilisation percentage, or None if not yet inspected.",
    )
    error_rate: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Errors per second observed, or None if not yet inspected.",
    )


# 2. IncidentDiagnosisAction

class IncidentDiagnosisAction(Action):
    """
    The atomic operation the agent submits at each environment step.

    Action types
    ------------
    query_logs       : Retrieve a subset of logs for `target` service.
                       Requires: target.
    inspect_service  : Pull full metrics for `target`.
                       Requires: target.
    check_dependency : Reveal health of the dependency link between two services.
                       Requires: target (service whose dependency to check).
    restart_service  : Attempt to restart `target` — modifies hidden state.
                       Requires: target.
    apply_patch      : Submit a JSON-encoded config patch to `target`.
                       Requires: target, patch_payload.
    propose_diagnosis: End the episode with a root-cause hypothesis.
                       Requires: diagnosis.
    """

    action_type: Literal[
        "query_logs",
        "inspect_service",
        "check_dependency",
        "restart_service",
        "apply_patch",
        "propose_diagnosis",
    ] = Field(..., description="The type of investigation or remediation action.")

    # Target service (required for most actions)
    target: Optional[str] = Field(
        default=None,
        description=(
            "Name of the service to act on. "
            "Required for: query_logs, inspect_service, check_dependency, "
            "restart_service, apply_patch."
        ),
    )

    # Optional log query filter
    query: Optional[str] = Field(
        default=None,
        description=(
            "Keyword filter for query_logs.  "
            "E.g. 'ERROR', 'timeout', 'OOM'.  Optional."
        ),
    )

    # Patch payload for apply_patch
    patch_payload: Optional[str] = Field(
        default=None,
        description=(
            "JSON-encoded string of config key-value pairs to apply. "
            "Required for: apply_patch."
        ),
    )

    # Root cause diagnosis string
    diagnosis: Optional[str] = Field(
        default=None,
        description=(
            "Root-cause hypothesis submitted by the agent. "
            "Required for: propose_diagnosis. "
            "Must match one of the valid root-cause IDs for the current task."
        ),
    )


# 3. IncidentDiagnosisObservation

class IncidentDiagnosisObservation(Observation):
    """
    Partial observation returned after every reset() and step() call.

    The agent can only see this — it cannot read the hidden SystemState.
    Information accumulates through investigative actions.
    """

    # Visible log lines (accumulate across query_logs calls)
    visible_logs: List[str] = Field(
        default_factory=list,
        description=(
            "Log lines retrieved so far via query_logs actions. "
            "Logs are partial and may contain noise."
        ),
    )

    # Current metric snapshots per service (populated by inspect_service)
    service_statuses: List[ServiceStatus] = Field(
        default_factory=list,
        description=(
            "Status snapshots for services the agent has inspected. "
            "Only services explicitly inspected appear here."
        ),
    )

    # Dependency health revealed by check_dependency actions
    dependency_health: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Map of 'source→target' dependency link to health info. "
            "Revealed incrementally via check_dependency actions."
        ),
    )

    # High-level alerts always visible (low-information surface signals)
    alerts: List[str] = Field(
        default_factory=list,
        description="System-level alert strings always visible, analogous to a pager alert.",
    )

    # (valid_diagnoses has been removed to prevent brute-force exploitation)

    # History of actions taken so far this episode
    previous_actions: List[str] = Field(
        default_factory=list,
        description="String summaries of all actions taken so far in this episode.",
    )

    # Step bookkeeping
    step_count: int = Field(
        default=0,
        ge=0,
        description="Number of steps taken so far in this episode.",
    )
    done: bool = Field(
        default=False,
        description="True if the episode has terminated.",
    )

    # Last step reward — always present so it survives WebSocket serialization
    reward: float = Field(
        default=0.0,
        description="Reward received for the last action (0.0 on reset).",
    )

    # Flexible diagnostic payload
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional info: last action result, reward, errors, etc.",
    )

    @field_validator("step_count")
    @classmethod
    def step_must_be_non_negative(cls, v: int) -> int:  # noqa: N805
        if v < 0:
            raise ValueError(f"step_count must be >= 0, got {v!r}")
        return v


# 4. IncidentDiagnosisReward

class IncidentDiagnosisReward(BaseModel):
    """
    Structured reward signal wrapping a scalar value.

    Wrapping the reward in a model is an OpenEnv spec requirement and makes
    it straightforward for logging pipelines to decompose the signal into its
    constituent parts without re-parsing episode logs.

    The scalar `value` is computed by the environment as:
        value = investigative_reward          # +0.02–0.22 per useful action
                + resolution_reward           # +0.99 (grader) on correct diagnosis/restart
                - step_cost                   # -0.02 per step
                - redundancy_penalty          # -0.05 on repeated queries
                - wrong_action_penalty        # -0.20 wrong restart, -0.30 wrong diagnosis

    Final episode grader score is strictly in (0.01, 0.99) — never exactly 0.0 or 1.0.
    """

    value: float = Field(
        ...,
        description="The scalar reward signal for the current step.",
    )
    action_type: str = Field(
        ...,
        description="The action_type that produced this reward.",
    )
    success: bool = Field(
        ...,
        description="True if the episode resolved successfully on this step.",
    )
    done: bool = Field(
        ...,
        description=(
            "True if the episode has ended — mirrors "
            "IncidentDiagnosisObservation.done for convenience."
        ),
    )
