"""
Incident Diagnosis Environment — Core Environment.

Class: IncidentDiagnosisEnvironment

Public methods
--------------
reset(task_id)   Load task JSON, initialise hidden state, return first observation.
step(action)     Execute one agent action, update state, compute reward, return observation.
state (property) Return State(episode_id, step_count) — for server bookkeeping.

Hidden State (never exposed)
----------------------------
The agent sees only partial observations — logs, metrics, dependency health —
revealed incrementally through investigative actions.

Reward Design (anti-hacking)
----------------------------
* Base step cost:    -0.02  (system downtime clock ticking)
* Redundant action:  -0.05  (re-querying same logs with same filter)
* Wrong restart:     -0.20  (restarting a service that is not the root cause)
* Wrong diagnosis:   -0.30  (incorrect propose_diagnosis)
* Correct diagnosis: +0.99  (easy/medium — pose is sufficient, grader score)
* Correct diagnosis + correct patch applied: +0.99 (hard task, grader score)
* Episode grader score strictly in (0.01, 0.99) — never exactly 0 or 1.

POMDP properties
----------------
* Agent never sees hidden_state directly.
* Logs are revealed one action at a time, max LOG_BATCH lines per call.
* Metrics only visible after inspect_service.
* Dependency health only visible after check_dependency.
* restart_service and apply_patch mutate hidden state.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        IncidentDiagnosisAction,
        IncidentDiagnosisObservation,
        ServiceStatus,
    )
except ImportError:
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from models import (  # type: ignore[no-redef]
        IncidentDiagnosisAction,
        IncidentDiagnosisObservation,
        ServiceStatus,
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_STEPS: int = 40
LOG_BATCH: int = 3          # lines revealed per query_logs call

TASKS_DIR = Path(__file__).parent.parent / "tasks"
TASK_FILE_MAP: Dict[str, Path] = {
    "easy":   TASKS_DIR / "task_easy.json",
    "medium": TASKS_DIR / "task_medium.json",
    "hard":   TASKS_DIR / "task_hard.json",
}

# Required action fields per action_type
_ACTION_REQUIRED: Dict[str, List[str]] = {
    "query_logs":        ["target"],
    "inspect_service":   ["target"],
    "check_dependency":  ["target"],
    "restart_service":   ["target"],
    "apply_patch":       ["target", "patch_payload"],
    "propose_diagnosis": ["diagnosis"],
}

# Noise lines added to logs based on failure mode
_NOISE_LOGS: Dict[str, List[str]] = {
    "service_crash": [
        "[NOISE] kernel: TCP connection keepalive probe failed (routine)",
        "[NOISE] systemd: service watchdog ping (routine)",
    ],
    "dependency_timeout": [
        "[NOISE] syslog: disk I/O stats collected (routine)",
        "[NOISE] prometheus: scrape interval normal (routine)",
        "[NOISE] cron: hourly job completed successfully",
    ],
    "cascading_hidden_failure": [
        "[NOISE] monitoring: false positive alert cleared",
        "[NOISE] deploybot: canary release 1% traffic nominal",
        "[NOISE] log_aggregator: buffer flush completed",
        "[NOISE] syslog: NTP sync OK",
        "[NOISE] kernel: network interface stats refreshed",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_summary(action: IncidentDiagnosisAction) -> str:
    """Return a compact human-readable summary of an action."""
    t = action.action_type
    if t == "query_logs":
        q = f" filter={action.query!r}" if action.query else ""
        return f"query_logs(target={action.target!r}{q})"
    if t == "inspect_service":
        return f"inspect_service(target={action.target!r})"
    if t == "check_dependency":
        return f"check_dependency(target={action.target!r})"
    if t == "restart_service":
        return f"restart_service(target={action.target!r})"
    if t == "apply_patch":
        return f"apply_patch(target={action.target!r}, payload={action.patch_payload!r})"
    if t == "propose_diagnosis":
        return f"propose_diagnosis(diagnosis={action.diagnosis!r})"
    return t


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class IncidentDiagnosisEnvironment(Environment):
    """
    POMDP-based incident diagnosis RL environment.

    An episode begins with reset(task_id) and proceeds via step(action) calls
    until done=True (correct diagnosis/patch, max steps, or fatal wrong action).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self._episode_id: str = str(uuid4())
        self.step_count: int = 0
        self._task_id: Optional[str] = None

        # Hidden ground truth (never serialised into observation)
        self._hidden: Dict[str, Any] = {}

        # Mutable runtime state
        self._service_statuses: Dict[str, str] = {}   # service -> status at runtime
        self._log_cursors: Dict[str, int] = {}         # service -> index into logs list
        self._revealed_deps: Dict[str, Any] = {}       # dep_key -> health info
        self._inspected: Set[str] = set()              # services fully inspected
        self._queried_log_keys: Set[str] = set()       # (service, query) pairs already done

        # Accumulated agent observation
        self._visible_logs: List[str] = []
        self._service_status_list: List[ServiceStatus] = []
        self._previous_actions: List[str] = []
        self._patch_applied: bool = False
        self._patch_correct: bool = False

        # Cumulative reward tracker
        self._cumulative_reward: float = 0.0

    # ── Public API ─────────────────────────────────────────────────────

    def reset(self, task_id: str = "easy") -> IncidentDiagnosisObservation:
        """Load task, initialise hidden state, return first observation."""
        if task_id not in TASK_FILE_MAP:
            raise ValueError(
                f"Unknown task_id {task_id!r}. Valid: {sorted(TASK_FILE_MAP)}"
            )

        task_path = TASK_FILE_MAP[task_id]
        with task_path.open() as fh:
            task_data = json.load(fh)

        self._episode_id = str(uuid4())
        self._task_id    = task_id
        self.step_count  = 0

        # Store full hidden state (deep-copy — never mutate task file data)
        self._hidden = copy.deepcopy(task_data)

        # Runtime service status mirrors hidden state initially
        self._service_statuses = {
            svc: info["status"]
            for svc, info in self._hidden["hidden_state"]["services"].items()
        }

        # Log cursors — start at 0 for each service
        self._log_cursors = {svc: 0 for svc in self._hidden["hidden_state"]["services"]}

        # Clear accumulated state
        self._revealed_deps     = {}
        self._inspected         = set()
        self._queried_log_keys  = set()
        self._visible_logs      = []
        self._service_status_list = []
        self._previous_actions  = []
        self._patch_applied     = False
        self._patch_correct     = False
        self._cumulative_reward = 0.0

        return self._make_obs(
            done=False,
            metadata={
                "task_id":    task_id,
                "episode_id": self._episode_id,
                "reward":     0.0,
                "message":    "Episode started. Investigate the incident.",
            },
        )

    def step(self, action: IncidentDiagnosisAction) -> IncidentDiagnosisObservation:
        """Execute one step. Returns observation, reward, done."""
        # 1. Validate required fields
        valid, field_err = self._validate_action_fields(action)
        if not valid:
            return self._make_obs(
                done=False,
                metadata={
                    "last_action_valid": False,
                    "error":   field_err,
                    "reward":  0.0,
                    "step":    self.step_count,
                },
            )

        # 2. Validate target exists (where applicable)
        if action.target is not None:
            if action.target not in self._hidden["hidden_state"]["services"]:
                self.step_count += 1
                return self._make_obs(
                    done=self.step_count >= self._hidden.get("max_steps", MAX_STEPS),
                    metadata={
                        "last_action_valid": False,
                        "error":  f"Unknown service: {action.target!r}. "
                                  f"Valid services: {list(self._hidden['hidden_state']['services'].keys())}",
                        "reward": -0.02,
                        "step":   self.step_count,
                    },
                )

        # 3. Dispatch action
        reward: float = -0.02   # base step cost
        done  : bool  = False
        meta  : Dict[str, Any] = {"last_action_valid": True}

        action_sum = _action_summary(action)
        self._previous_actions.append(action_sum)

        t = action.action_type

        if t == "query_logs":
            reward, meta = self._handle_query_logs(action, reward, meta)

        elif t == "inspect_service":
            reward, meta = self._handle_inspect_service(action, reward, meta)

        elif t == "check_dependency":
            reward, meta = self._handle_check_dependency(action, reward, meta)

        elif t == "restart_service":
            reward, meta, done = self._handle_restart_service(action, reward, meta)

        elif t == "apply_patch":
            reward, meta, done = self._handle_apply_patch(action, reward, meta)

        elif t == "propose_diagnosis":
            reward, meta, done = self._handle_propose_diagnosis(action, reward, meta)

        # 4. Increment step count; enforce max steps
        self.step_count += 1
        max_limit = self._hidden.get("max_steps", MAX_STEPS)
        if not done and self.step_count >= max_limit:
            done = True
            meta["timeout"] = True
            meta["message"] = "Max steps reached without correct diagnosis."
            reward -= 0.10   # timeout penalty

        self._cumulative_reward += reward
        meta["reward"] = reward
        meta["step"]   = self.step_count
        meta["cumulative_reward"] = round(self._cumulative_reward, 4)

        return self._make_obs(done=done, metadata=meta)

    @property
    def state(self) -> State:
        return State(episode_id=self._episode_id, step_count=self.step_count)

    # ── Action handlers ────────────────────────────────────────────────

    def _handle_query_logs(
        self,
        action: IncidentDiagnosisAction,
        reward: float,
        meta: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Reveal LOG_BATCH log lines from `target`, optionally filtered by query."""
        svc    = action.target
        query  = (action.query or "").lower().strip()
        key    = (svc, query)

        if key in self._queried_log_keys:
            # Redundant query penalised
            reward -= 0.05
            meta["message"] = (
                f"Already queried logs for {svc!r} with filter {query!r}. "
                "No new information."
            )
            meta["redundant"] = True
            return reward, meta

        self._queried_log_keys.add(key)

        service_data = self._hidden["hidden_state"]["services"][svc]
        all_logs: List[str] = list(service_data["logs"])

        # Add noise lines based on failure mode
        failure_mode = self._hidden["hidden_state"]["failure_mode"]
        noise = _NOISE_LOGS.get(failure_mode, [])
        # Insert noise randomly
        enriched = list(all_logs)
        for i, n in enumerate(noise):
            pos = (i * 2 + 1) % max(len(enriched), 1)
            enriched.insert(pos, n)

        # Apply keyword filter if given
        if query:
            enriched = [ln for ln in enriched if query in ln.lower()]
            if not enriched:
                enriched = [f"[INFO] No log entries matching {query!r} found for {svc}."]

        # Pull a batch
        cursor = self._log_cursors.get(svc, 0)
        batch  = enriched[cursor: cursor + LOG_BATCH]
        self._log_cursors[svc] = cursor + len(batch)

        if not batch:
            batch = [f"[INFO] No more logs for {svc}."]
            reward -= 0.03   # slight penalty for exhausted log source

        self._visible_logs.extend(batch)

        # Positive signal only if useful (non-noise) content revealed
        useful = [ln for ln in batch if "[NOISE]" not in ln]
        if useful:
            reward += 0.05   # small positive, can't exploit without bound

        meta["logs_revealed"] = batch
        meta["message"] = f"Revealed {len(batch)} log line(s) from {svc!r}."
        return reward, meta

    def _handle_inspect_service(
        self,
        action: IncidentDiagnosisAction,
        reward: float,
        meta: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Reveal full metrics for `target` service."""
        svc = action.target

        if svc in self._inspected:
            reward -= 0.05
            meta["message"] = f"Already inspected {svc!r}. No new information."
            meta["redundant"] = True
            return reward, meta

        self._inspected.add(svc)
        svc_data = self._hidden["hidden_state"]["services"][svc]
        runtime_status = self._service_statuses.get(svc, svc_data["status"])

        status_obj = ServiceStatus(
            name       = svc,
            status     = runtime_status,          # type: ignore[arg-type]
            cpu_pct    = svc_data.get("cpu_pct"),
            mem_pct    = svc_data.get("mem_pct"),
            error_rate = svc_data.get("error_rate"),
        )

        # Update or append service in list
        self._service_status_list = [
            s for s in self._service_status_list if s.name != svc
        ]
        self._service_status_list.append(status_obj)

        # Reward signal — informative if service is unhealthy
        if runtime_status in ("degraded", "down"):
            reward += 0.10
        else:
            reward += 0.02   # still useful but less so for healthy service

        meta["service_status"] = status_obj.model_dump()
        meta["message"] = f"Inspected {svc!r}: status={runtime_status}."
        return reward, meta

    def _handle_check_dependency(
        self,
        action: IncidentDiagnosisAction,
        reward: float,
        meta: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Reveal all dependency health links for `target`."""
        svc = action.target
        deps_revealed: Dict[str, Any] = {}

        dep_health_map = self._hidden["hidden_state"].get("dependency_health", {})

        newly_revealed = False
        for link_key, health_info in dep_health_map.items():
            # e.g. "api_gateway->auth_service"
            src, _, tgt = link_key.partition("->")
            # Only reveal if svc is the SOURCE (outgoing links)
            if src == svc:
                if link_key not in self._revealed_deps:
                    self._revealed_deps[link_key] = health_info
                    deps_revealed[link_key] = health_info
                    newly_revealed = True

        if not newly_revealed:
            reward -= 0.05
            meta["message"] = (
                f"All dependency links for {svc!r} already known. No new information."
            )
            meta["redundant"] = True
            return reward, meta

        if not deps_revealed:
            meta["message"] = f"No dependency links found for {svc!r}."
            return reward, meta

        # Reward for revealing unhealthy links
        unhealthy = sum(
            1 for h in deps_revealed.values()
            if not h.get("healthy", True)
        )
        reward += 0.08 * unhealthy

        meta["dependency_health_revealed"] = deps_revealed
        meta["message"] = (
            f"Revealed {len(deps_revealed)} dependency link(s) for {svc!r}."
        )
        return reward, meta

    def _handle_restart_service(
        self,
        action: IncidentDiagnosisAction,
        reward: float,
        meta: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any], bool]:
        """Restart `target` — heals if it's in restart_heals, penalised otherwise."""
        svc = action.target
        root_cause   = self._hidden["hidden_state"]["root_cause"]
        restart_heals: List[str] = self._hidden.get("restart_heals", [])

        if svc in restart_heals:
            # Correct restart — heal the service
            self._service_statuses[svc] = "healthy"
            # Check if this full heals the episode (easy task)
            if self._task_id == "easy":
                # Cascade heal: auth_service recovers once database is up
                for dependent in self._get_dependents(svc):
                    self._service_statuses[dependent] = "healthy"

                reward += 1.0
                meta["message"] = (
                    f"Restarted {svc!r} — root cause resolved! "
                    "System recovering. Episode complete."
                )
                meta["success"] = True
                return reward, meta, True

            # Partial heal — service back but root cause remains
            reward += 0.30
            meta["message"] = (
                f"Restarted {svc!r} — service recovered. "
                "Root cause may still be present."
            )

        else:
            # Wrong restart — penalise
            reward -= 0.20
            meta["message"] = (
                f"Restarted {svc!r} — no improvement. "
                f"The root cause is not a simple crash of {svc!r}. "
                "Service resumed but underlying issue persists."
            )

        return reward, meta, False

    def _handle_apply_patch(
        self,
        action: IncidentDiagnosisAction,
        reward: float,
        meta: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any], bool]:
        """Apply a config patch to `target`. Required for 'hard' task resolution."""
        svc = action.target
        patch_heals: Dict[str, Any] = self._hidden.get("patch_heals", {})
        required_patch_target = self._hidden.get("required_patch_target")
        required_patch_keys: List[str] = self._hidden.get("required_patch_keys", [])

        # Parse patch payload
        try:
            patch = json.loads(action.patch_payload or "{}")
        except (json.JSONDecodeError, TypeError):
            reward -= 0.10
            meta["last_action_valid"] = False
            meta["error"] = "patch_payload must be valid JSON."
            meta["message"] = "Invalid patch payload — must be a JSON object string."
            return reward, meta, False

        if svc not in patch_heals:
            reward -= 0.20
            meta["message"] = (
                f"Applying patch to {svc!r} has no effect. "
                "This is not the service that requires configuration correction."
            )
            return reward, meta, False

        expected = patch_heals[svc]

        # Check if submitted patch covers the required keys
        correct_keys = [k for k in required_patch_keys if k in patch]

        if required_patch_target and svc == required_patch_target and correct_keys:
            # Validate values — at least the required keys must be present
            self._patch_applied = True
            self._patch_correct = True
            self._service_statuses[svc] = "healthy"
            # Cascade heals
            for dependent in self._get_dependents(svc):
                self._service_statuses[dependent] = "healthy"

            # Patch is correct — but only complete if diagnosis was also correct
            # (for hard task the environment issues partial reward until propose_diagnosis)
            reward += 0.50
            meta["patch_accepted"] = True
            meta["message"] = (
                f"Patch applied to {svc!r} — correct keys accepted. "
                "Service configuration corrected. Now propose your diagnosis."
            )
            return reward, meta, False

        else:
            # Wrong patch content
            reward -= 0.10
            meta["patch_accepted"] = False
            meta["message"] = (
                f"Patch applied to {svc!r} but key(s) {required_patch_keys} "
                "are missing or incorrect."
            )
            return reward, meta, False

    def _handle_propose_diagnosis(
        self,
        action: IncidentDiagnosisAction,
        reward: float,
        meta: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any], bool]:
        """Evaluate the agent's diagnosis. Ends the episode."""
        diagnosis    = (action.diagnosis or "").strip().lower()
        correct      = self._hidden["hidden_state"]["root_cause"].lower()
        valid_diags  = self._hidden.get("valid_diagnoses", [])
        
        # We perform loose matching by normalising spaces to underscores
        normalized_diag = diagnosis.replace(" ", "_")
        
        if correct in normalized_diag or correct.replace("_", " ") in diagnosis:
            # For hard task, also require correct patch
            if self._task_id == "hard" and not self._patch_correct:
                reward += 0.20   # partial credit for correct root-cause ID
                meta["message"] = (
                    f"Correct root cause identified: {diagnosis!r}. "
                    "But you must also apply a correct patch to close the incident. "
                    "Use apply_patch on the correct service."
                )
                meta["diagnosis_correct"] = True
                meta["patch_required"]    = True
                return reward, meta, False   # not done yet

            reward += 1.0
            meta["success"]    = True
            meta["message"]    = (
                f"Correct! Root cause: {correct!r}. "
                f"Resolved in {self.step_count + 1} step(s)."
            )
            meta["diagnosis_correct"] = True
        else:
            reward -= 0.30
            meta["success"] = False
            meta["message"] = (
                f"Incorrect diagnosis: {diagnosis!r}. "
                f"The actual root cause was {correct!r}. "
                "Episode ended."
            )
            meta["diagnosis_correct"] = False

        return reward, meta, True   # ANY diagnosis result ends the episode (except partial hard)

    # ── Helpers ────────────────────────────────────────────────────────

    def _validate_action_fields(
        self, action: IncidentDiagnosisAction
    ) -> Tuple[bool, str]:
        required = _ACTION_REQUIRED.get(action.action_type, [])
        missing = [
            f for f in required
            if getattr(action, f, None) is None or getattr(action, f, None) == ""
        ]
        if missing:
            return False, (
                f"Action '{action.action_type}' missing required fields: {missing}"
            )
        return True, ""

    def _get_dependents(self, svc: str) -> List[str]:
        """Return services that depend on `svc` (reversed dependency map)."""
        deps_map: Dict[str, List[str]] = self._hidden.get("dependencies", {})
        return [
            dependent
            for dependent, deps in deps_map.items()
            if svc in deps
        ]

    def _make_obs(
        self,
        done: bool,
        metadata: Dict[str, Any],
    ) -> IncidentDiagnosisObservation:
        """Build a partial observation — hidden state is never included."""
        reward_val = float(metadata.get("reward", 0.0))
        return IncidentDiagnosisObservation(
            visible_logs      = list(self._visible_logs),
            service_statuses  = list(self._service_status_list),
            dependency_health = dict(self._revealed_deps),
            alerts            = list(self._hidden.get("initial_alerts", [])),
            previous_actions  = list(self._previous_actions),
            step_count        = self.step_count,
            done              = done,
            reward            = reward_val,   # surfaced directly so client can read it
            metadata          = metadata,
        )


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = IncidentDiagnosisEnvironment()

    print("=" * 60)
    print("  Smoke test: easy task")
    print("=" * 60)
    obs = env.reset(task_id="easy")
    print(f"  reset → step={obs.step_count}  done={obs.done}")
    print(f"  alerts: {obs.alerts}")
    print(f"  metadata keys: {list(obs.metadata.keys())}")

    steps = [
        IncidentDiagnosisAction(action_type="inspect_service",  target="database"),
        IncidentDiagnosisAction(action_type="query_logs",       target="database"),
        IncidentDiagnosisAction(action_type="propose_diagnosis", diagnosis="database_crash"),
    ]
    for act in steps:
        result = env.step(act)
        r = result.metadata.get("reward", 0.0)
        print(
            f"  step {result.step_count:2d}  {act.action_type}  "
            f"reward={r:+.3f}  done={result.done}  "
            f"msg={result.metadata.get('message', '')}"
        )
        if result.done:
            print(f"  → success={result.metadata.get('success', False)}")
            break

    print("\n" + "=" * 60)
    print("  Smoke test: hard task (partial path)")
    print("=" * 60)
    obs = env.reset(task_id="hard")
    print(f"  reset → step={obs.step_count}")
    hard_steps = [
        IncidentDiagnosisAction(action_type="check_dependency",  target="user_service"),
        IncidentDiagnosisAction(action_type="query_logs",         target="routing_config"),
        IncidentDiagnosisAction(action_type="apply_patch",        target="routing_config",
                                patch_payload='{"eu-west-3": "https://svc.eu-west-3.internal:8080"}'),
        IncidentDiagnosisAction(action_type="propose_diagnosis",  diagnosis="routing_config_misconfiguration"),
    ]
    for act in hard_steps:
        result = env.step(act)
        r = result.metadata.get("reward", 0.0)
        print(
            f"  step {result.step_count:2d}  {act.action_type}  "
            f"reward={r:+.3f}  done={result.done}  "
            f"msg={result.metadata.get('message', '')}"
        )
        if result.done:
            print(f"  → success={result.metadata.get('success', False)}")
            break
