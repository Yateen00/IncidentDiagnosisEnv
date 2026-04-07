#!/usr/bin/env python3
"""
Standalone validator for Incident Diagnosis Environment task JSON files.

Run from anywhere:
    uv run python validate_tasks.py

Checks performed per task file:
  1. File exists and is valid JSON.
  2. Top-level keys include all required fields.
  3. hidden_state contains root_cause, failure_mode, services.
  4. Each service in hidden_state.services conforms to the service schema.
  5. Dependencies map references only known services.
  6. No circular dependencies exist (topological sort check).
  7. valid_diagnoses is a non-empty list of strings.
  8. correct_diagnosis is present in valid_diagnoses.
  9. initial_alerts is a non-empty list of strings.
 10. restart_heals references only known services.
 11. patch_heals (if present) references only known services.
 12. Hard task: required_patch_target and required_patch_keys are consistent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# ── locate tasks dir relative to this script ──────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
TASKS_DIR  = SCRIPT_DIR / "tasks"
TASK_FILES = ["task_easy.json", "task_medium.json", "task_hard.json"]

REQUIRED_TOP_LEVEL = {
    "task_id", "description", "services", "dependencies",
    "hidden_state", "initial_alerts", "valid_diagnoses",
    "correct_diagnosis", "restart_heals", "patch_heals",
}
REQUIRED_HIDDEN_STATE = {"root_cause", "failure_mode", "services", "dependency_health"}
REQUIRED_SERVICE_KEYS = {"status", "cpu_pct", "mem_pct", "error_rate", "logs"}
VALID_STATUSES = {"healthy", "degraded", "down", "unknown"}

ERRORS:   list[str] = []
WARNINGS: list[str] = []


# ── helpers ────────────────────────────────────────────────────────────────────

def err(task: str, msg: str) -> None:
    ERRORS.append(f"[{task}] ERROR: {msg}")


def warn(task: str, msg: str) -> None:
    WARNINGS.append(f"[{task}] WARN:  {msg}")


def ok(msg: str) -> None:
    print(f"  ✓  {msg}")


# ── check 1: valid JSON ────────────────────────────────────────────────────────

def load_json(path: Path, task: str) -> dict | None:
    if not path.exists():
        err(task, f"File not found: {path}")
        return None
    try:
        with path.open() as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        err(task, f"Invalid JSON: {e}")
        return None
    ok(f"Valid JSON ({path.stat().st_size} bytes)")
    return data


# ── check 2: top-level keys ────────────────────────────────────────────────────

def check_top_level(data: dict, task: str) -> bool:
    keys    = set(data.keys())
    missing = REQUIRED_TOP_LEVEL - keys
    valid   = True
    if missing:
        err(task, f"Missing top-level keys: {missing}")
        valid = False
    if valid:
        ok(f"Top-level keys: all {len(REQUIRED_TOP_LEVEL)} required fields present")
    return valid


# ── check 3: hidden_state structure ───────────────────────────────────────────

def check_hidden_state(data: dict, task: str) -> bool:
    hs = data.get("hidden_state", {})
    if not isinstance(hs, dict):
        err(task, "hidden_state must be a dict")
        return False

    missing = REQUIRED_HIDDEN_STATE - set(hs.keys())
    if missing:
        err(task, f"hidden_state missing keys: {missing}")
        return False

    ok("hidden_state structure: root_cause, failure_mode, services, dependency_health present")
    return True


# ── check 4: service schema ────────────────────────────────────────────────────

def check_services(data: dict, task: str) -> bool:
    known_services = set(data.get("services", []))
    hs_services    = data.get("hidden_state", {}).get("services", {})

    if not isinstance(hs_services, dict) or len(hs_services) == 0:
        err(task, "hidden_state.services must be a non-empty dict")
        return False

    valid = True

    # Check top-level services list matches hidden_state.services
    hs_names = set(hs_services.keys())
    if known_services != hs_names:
        only_list = known_services - hs_names
        only_hs   = hs_names - known_services
        if only_list:
            warn(task, f"Services in 'services' list but not in hidden_state.services: {only_list}")
        if only_hs:
            warn(task, f"Services in hidden_state.services but not in 'services' list: {only_hs}")

    for svc_name, svc in hs_services.items():
        if not isinstance(svc, dict):
            err(task, f"hidden_state.services.{svc_name}: must be a dict")
            valid = False
            continue

        missing = REQUIRED_SERVICE_KEYS - set(svc.keys())
        if missing:
            err(task, f"hidden_state.services.{svc_name}: missing keys {missing}")
            valid = False

        if "status" in svc and svc["status"] not in VALID_STATUSES:
            err(task, f"hidden_state.services.{svc_name}.status: "
                f"invalid value {svc['status']!r}. Must be one of {VALID_STATUSES}")
            valid = False

        if "logs" in svc:
            if not isinstance(svc["logs"], list):
                err(task, f"hidden_state.services.{svc_name}.logs must be a list")
                valid = False
            elif len(svc["logs"]) == 0:
                warn(task, f"hidden_state.services.{svc_name}.logs is empty")
            elif not all(isinstance(ln, str) for ln in svc["logs"]):
                err(task, f"hidden_state.services.{svc_name}.logs must be list of strings")
                valid = False

    if valid:
        ok(f"hidden_state.services: {len(hs_services)} service(s) conform to schema")
    return valid


# ── check 5: dependency references ────────────────────────────────────────────

def check_dependency_refs(data: dict, task: str) -> bool:
    known_services = set(data.get("services", []))
    dep_map: dict  = data.get("dependencies", {})

    if not isinstance(dep_map, dict):
        err(task, "dependencies must be a dict mapping service → list of services")
        return False

    valid = True
    for svc, deps in dep_map.items():
        if svc not in known_services:
            err(task, f"dependencies: key '{svc}' not in services list")
            valid = False
        if not isinstance(deps, list):
            err(task, f"dependencies.{svc}: must be a list")
            valid = False
            continue
        for dep in deps:
            if dep not in known_services:
                err(task, f"dependencies.{svc}: dependency '{dep}' not in services list")
                valid = False

    if valid:
        ok(f"dependencies: all {len(dep_map)} service dependency refs resolve")
    return valid


# ── check 6: no circular dependencies ─────────────────────────────────────────

def check_no_cycles(data: dict, task: str) -> bool:
    dep_map: dict  = data.get("dependencies", {})
    visited:   set[str] = set()
    in_progress: set[str] = set()

    def dfs(node: str) -> bool:
        if node in in_progress:
            return False  # cycle
        if node in visited:
            return True
        in_progress.add(node)
        for dep in dep_map.get(node, []):
            if not dfs(dep):
                return False
        in_progress.discard(node)
        visited.add(node)
        return True

    for svc in dep_map:
        if not dfs(svc):
            err(task, f"dependencies: circular dependency detected involving '{svc}'")
            return False

    ok("dependencies: no circular dependencies")
    return True


# ── check 7 & 8: diagnoses ────────────────────────────────────────────────────

def check_diagnoses(data: dict, task: str) -> bool:
    valid_diags  = data.get("valid_diagnoses", [])
    correct_diag = data.get("correct_diagnosis", None)
    valid        = True

    if not isinstance(valid_diags, list) or len(valid_diags) == 0:
        err(task, "valid_diagnoses must be a non-empty list")
        valid = False
    elif not all(isinstance(d, str) for d in valid_diags):
        err(task, "valid_diagnoses must be a list of strings")
        valid = False

    if correct_diag is None:
        err(task, "correct_diagnosis is missing")
        valid = False
    elif not isinstance(correct_diag, str):
        err(task, "correct_diagnosis must be a string")
        valid = False
    elif correct_diag not in valid_diags:
        err(task, f"correct_diagnosis {correct_diag!r} is not in valid_diagnoses: {valid_diags}")
        valid = False

    if valid:
        ok(f"diagnoses: correct_diagnosis={correct_diag!r} present in {len(valid_diags)} valid options")
    return valid


# ── check 9: alerts ───────────────────────────────────────────────────────────

def check_alerts(data: dict, task: str) -> bool:
    alerts = data.get("initial_alerts", [])
    if not isinstance(alerts, list) or len(alerts) == 0:
        err(task, "initial_alerts must be a non-empty list")
        return False
    if not all(isinstance(a, str) for a in alerts):
        err(task, "initial_alerts must be a list of strings")
        return False
    ok(f"initial_alerts: {len(alerts)} alert(s) present")
    return True


# ── check 10 & 11: restart_heals and patch_heals ──────────────────────────────

def check_heals(data: dict, task: str) -> bool:
    known = set(data.get("services", []))
    valid = True

    restart_heals = data.get("restart_heals", [])
    if not isinstance(restart_heals, list):
        err(task, "restart_heals must be a list")
        valid = False
    else:
        for svc in restart_heals:
            if svc not in known:
                err(task, f"restart_heals: '{svc}' not in services list")
                valid = False

    patch_heals = data.get("patch_heals", {})
    if not isinstance(patch_heals, dict):
        err(task, "patch_heals must be a dict")
        valid = False
    else:
        for svc in patch_heals:
            if svc not in known:
                err(task, f"patch_heals: '{svc}' not in services list")
                valid = False

    if valid:
        ok(f"heals: restart_heals={restart_heals}, patch_heals targets={list(patch_heals.keys())}")
    return valid


# ── check 12: hard task patch consistency ─────────────────────────────────────

def check_hard_task_patch(data: dict, task: str) -> None:
    """Only runs if task has required_patch_target set."""
    rpt  = data.get("required_patch_target")
    rpk  = data.get("required_patch_keys", [])
    ph   = data.get("patch_heals", {})
    known = set(data.get("services", []))

    if rpt is None:
        return  # not a patch-required task

    if rpt not in known:
        err(task, f"required_patch_target '{rpt}' not in services list")
        return
    if rpt not in ph:
        err(task, f"required_patch_target '{rpt}' not in patch_heals")
        return
    if not isinstance(rpk, list) or len(rpk) == 0:
        err(task, "required_patch_keys must be a non-empty list")
        return
    for key in rpk:
        if key not in ph.get(rpt, {}):
            err(task, f"required_patch_keys[{key!r}] not found in patch_heals[{rpt!r}]")
            return

    ok(f"hard task patch: required_patch_target={rpt!r}, required_patch_keys={rpk}")


# ── diagnosis vs hidden_state root_cause ──────────────────────────────────────

def check_root_cause_consistency(data: dict, task: str) -> None:
    root_cause   = data.get("hidden_state", {}).get("root_cause")
    correct_diag = data.get("correct_diagnosis")

    if root_cause and correct_diag and root_cause != correct_diag:
        err(task,
            f"hidden_state.root_cause={root_cause!r} != correct_diagnosis={correct_diag!r}. "
            "These must match — the evaluator uses correct_diagnosis to score.")
    elif root_cause and correct_diag:
        ok(f"root_cause/correct_diagnosis consistency: {root_cause!r}")


# ── main ───────────────────────────────────────────────────────────────────────

def validate_task(filename: str) -> None:
    task = filename.replace(".json", "")
    path = TASKS_DIR / filename

    print(f"\n{'='*60}")
    print(f"  Validating {filename}")
    print(f"{'='*60}")

    data = load_json(path, task)
    if data is None:
        return

    if not check_top_level(data, task):
        print("  (skipping remaining checks — top-level keys invalid)")
        return

    check_hidden_state(data, task)
    check_services(data, task)
    check_dependency_refs(data, task)
    check_no_cycles(data, task)
    check_diagnoses(data, task)
    check_alerts(data, task)
    check_heals(data, task)
    check_hard_task_patch(data, task)
    check_root_cause_consistency(data, task)


def main() -> None:
    print("\nIncident Diagnosis Env — Task JSON Validator")
    print(f"Tasks directory: {TASKS_DIR}")

    if not TASKS_DIR.exists():
        print(f"\nFATAL: tasks directory not found at {TASKS_DIR}", file=sys.stderr)
        sys.exit(1)

    for filename in TASK_FILES:
        validate_task(filename)

    print(f"\n{'='*60}")

    if WARNINGS:
        print(f"\n⚠  {len(WARNINGS)} warning(s):")
        for w in WARNINGS:
            print(f"   {w}")

    if ERRORS:
        print(f"\n✗  {len(ERRORS)} error(s):")
        for e in ERRORS:
            print(f"   {e}")
        print()
        sys.exit(1)
    else:
        print("\n✓  All task files passed validation.")
        if WARNINGS:
            print("   (review warnings above before proceeding)")
        print()


if __name__ == "__main__":
    main()
