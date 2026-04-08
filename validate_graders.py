#!/usr/bin/env python3
"""Validate deterministic task graders.

This script builds fixed synthetic trajectories and verifies that:
- every grader returns a score in [0.0, 1.0]
- scores are deterministic across repeated calls
"""

from __future__ import annotations

from graders import grade_task


def _assert_deterministic(task_id: str, trajectory: list[dict]) -> None:
    s1 = grade_task(task_id, trajectory)
    s2 = grade_task(task_id, trajectory)
    if s1 != s2:
        raise AssertionError(f"{task_id}: grader is non-deterministic ({s1} != {s2})")
    if not (0.0 <= s1 <= 1.0):
        raise AssertionError(f"{task_id}: score out of range [0,1]: {s1}")
    print(f"{task_id:<6} score={s1:.4f} deterministic=ok")


def main() -> None:
    easy_traj = [
        {
            "action": {"action_type": "inspect_service", "target": "database"},
            "observation": {"metadata": {}},
        },
        {
            "action": {"action_type": "propose_diagnosis", "diagnosis": "database_crash"},
            "observation": {"metadata": {"success": True}},
        },
    ]

    medium_traj = [
        {
            "action": {"action_type": "check_dependency", "target": "payment_service"},
            "observation": {"metadata": {}},
        },
        {
            "action": {"action_type": "inspect_service", "target": "cache"},
            "observation": {"metadata": {}},
        },
        {
            "action": {"action_type": "propose_diagnosis", "diagnosis": "cache_memory_exhaustion"},
            "observation": {"metadata": {"success": True}},
        },
    ]

    hard_partial = [
        {
            "action": {"action_type": "query_logs", "target": "routing_config"},
            "observation": {"metadata": {}},
        },
        {
            "action": {"action_type": "propose_diagnosis", "diagnosis": "routing_config_misconfiguration"},
            "observation": {"metadata": {"success": False}},
        },
    ]

    hard_full = [
        {
            "action": {"action_type": "apply_patch", "target": "routing_config"},
            "observation": {"metadata": {"patch_accepted": True}},
        },
        {
            "action": {"action_type": "propose_diagnosis", "diagnosis": "routing_config_misconfiguration"},
            "observation": {"metadata": {"success": True}},
        },
    ]

    _assert_deterministic("easy", easy_traj)
    _assert_deterministic("medium", medium_traj)
    _assert_deterministic("hard", hard_partial)
    _assert_deterministic("hard", hard_full)
    print("all graders validated")


if __name__ == "__main__":
    main()
