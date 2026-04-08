"""Deterministic task graders for IncidentDiagnosisEnv.

Each grader maps an episode trajectory into a normalized score in [0.0, 1.0].
The logic is deterministic and only depends on the provided trajectory payload.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


TrajectoryStep = Dict[str, Any]
Trajectory = List[TrajectoryStep]


def _clamp01(value: float) -> float:
    return max(0.01, min(0.99, round(value, 4)))


def _norm(text: Optional[str]) -> str:
    return (text or "").strip().lower().replace(" ", "_")


def _action_type(step: TrajectoryStep) -> str:
    action = step.get("action", {}) or {}
    if isinstance(action, dict):
        return str(action.get("action_type", "")).strip().lower()
    return ""


def _action_target(step: TrajectoryStep) -> str:
    action = step.get("action", {}) or {}
    if isinstance(action, dict):
        return str(action.get("target", "")).strip().lower()
    return ""


def _action_diag(step: TrajectoryStep) -> str:
    action = step.get("action", {}) or {}
    if isinstance(action, dict):
        return _norm(action.get("diagnosis"))
    return ""


def _action_patch_payload(step: TrajectoryStep) -> Dict[str, Any]:
    action = step.get("action", {}) or {}
    if not isinstance(action, dict):
        return {}
    raw = action.get("patch_payload")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            import json

            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _meta(step: TrajectoryStep) -> Dict[str, Any]:
    obs = step.get("observation", {}) or {}
    if isinstance(obs, dict):
        meta = obs.get("metadata", {}) or {}
        if isinstance(meta, dict):
            return meta
    return {}


def _final_meta(trajectory: Trajectory) -> Dict[str, Any]:
    return _meta(trajectory[-1]) if trajectory else {}


def _investigation_bonus(trajectory: Trajectory, cap: float = 0.2) -> float:
    """Reward information gathering diversity, capped to keep final objective dominant."""
    seen = set()
    for step in trajectory:
        at = _action_type(step)
        if at in {"query_logs", "inspect_service", "check_dependency"}:
            seen.add(at)
    return min(cap, 0.07 * len(seen))


def grade_easy(trajectory: Trajectory) -> float:
    """Easy task grading.

    1.0: Correct diagnosis or successful root-cause restart.
    Partial: Investigation progress + some reward shaping.
    """
    if not trajectory:
        return 0.01

    final = _final_meta(trajectory)
    if bool(final.get("success", False)):
        return 0.99

    diagnosed = any(
        _action_type(step) == "propose_diagnosis" and "database_crash" in _action_diag(step)
        for step in trajectory
    )
    restarted_db = any(
        _action_type(step) == "restart_service" and _action_target(step) == "database"
        for step in trajectory
    )

    # Easy objective is to identify/resolve the root cause; either is full credit.
    if diagnosed or restarted_db:
        return 0.99

    score = 0.0
    score += _investigation_bonus(trajectory, cap=0.2)
    return _clamp01(score)


def grade_medium(trajectory: Trajectory) -> float:
    """Medium task grading.

    1.0: Correct diagnosis of cache memory exhaustion.
    Partial: Agent investigated cache path and collected dependency evidence.
    """
    if not trajectory:
        return 0.01

    final = _final_meta(trajectory)
    if bool(final.get("success", False)):
        return 0.99

    diagnosed = any(
        _action_type(step) == "propose_diagnosis" and "cache_memory_exhaustion" in _action_diag(step)
        for step in trajectory
    )
    inspected_cache = any(
        _action_type(step) == "inspect_service" and _action_target(step) == "cache"
        for step in trajectory
    )
    checked_cache_deps = any(
        _action_type(step) == "check_dependency" and _action_target(step) in {"payment_service", "inventory_service", "billing_service"}
        for step in trajectory
    )

    score = 0.0
    if diagnosed:
        score += 0.75
    if inspected_cache:
        score += 0.1
    if checked_cache_deps:
        score += 0.1
    score += _investigation_bonus(trajectory, cap=0.15)
    return _clamp01(score)


def grade_hard(trajectory: Trajectory) -> float:
    """Hard task grading.

    1.0: Correct diagnosis + correct patch accepted.
    0.5: Correct diagnosis but missing/incorrect patch.
    0.2-0.4: strong investigation progress without full resolution.
    """
    if not trajectory:
        return 0.01

    final = _final_meta(trajectory)
    if bool(final.get("success", False)):
        return 0.99

    diagnosed = any(
        _action_type(step) == "propose_diagnosis" and "routing_config_misconfiguration" in _action_diag(step)
        for step in trajectory
    )
    patch_accepted_meta = any(bool(_meta(step).get("patch_accepted", False)) for step in trajectory)
    patch_from_action = any(
        _action_type(step) == "apply_patch"
        and _action_target(step) == "routing_config"
        and "eu-west-3" in _action_patch_payload(step)
        for step in trajectory
    )
    patch_accepted = patch_accepted_meta or patch_from_action
    touched_routing = any(
        _action_target(step) == "routing_config"
        and _action_type(step) in {"query_logs", "inspect_service", "check_dependency", "apply_patch"}
        for step in trajectory
    )

    if diagnosed and patch_accepted:
        return 0.99
    if diagnosed and not patch_accepted:
        return 0.5

    score = 0.0
    if touched_routing:
        score += 0.2
    score += _investigation_bonus(trajectory, cap=0.2)
    return _clamp01(score)


def grade_task(task_id: str, trajectory: Trajectory) -> float:
    """Dispatch to the deterministic grader for a task."""
    task = _norm(task_id)
    if task == "easy":
        return grade_easy(trajectory)
    if task == "medium":
        return grade_medium(trajectory)
    if task == "hard":
        return grade_hard(trajectory)
    raise ValueError(f"Unknown task_id {task_id!r}. Expected one of: easy, medium, hard")
