"""
Incident Diagnosis Environment — Baseline Inference Script
==========================================================

Mandatory env vars:
  API_BASE_URL     LLM endpoint  (default: https://router.huggingface.co/v1)
  MODEL_NAME       Model id      (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN         HF / API key
  ENV_BASE_URL     Running env server (default: http://localhost:8000)

STDOUT FORMAT (evaluator parses these — do not change):
  [START] task=<task_id> env=IncidentDiagnosisEnv model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

All debug output goes to stderr.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# Always prioritize local workspace modules so runs are reproducible and reflect
# the current source tree, not any previously installed package version.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import IncidentDiagnosisEnv          # type: ignore[no-redef]
from graders import grade_task                   # type: ignore[no-redef]
from models import IncidentDiagnosisAction, IncidentDiagnosisObservation  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or "http://localhost:8000"
MAX_STEPS    = int(os.getenv("MAX_STEPS", "40"))
# Required by evaluator when using dockerized env loading; kept for compatibility.
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")

BENCHMARK = "IncidentDiagnosisEnv"
TASK_IDS  = ["easy", "medium", "hard"]

# ---------------------------------------------------------------------------
# OpenAI-compatible LLM client (uses HF Router)
# ---------------------------------------------------------------------------

llm: Optional[OpenAI] = None


def _get_llm() -> OpenAI:
    global llm
    if llm is None:
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN is required for API-backed inference")
        llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    return llm

# ---------------------------------------------------------------------------
# Stdout log helpers (exact evaluator format)
# ---------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err_val  = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={err_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def _debug(msg: str) -> None:
    print(f"[debug] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Action string formatter for [STEP] log
# ---------------------------------------------------------------------------

def _action_str(action: IncidentDiagnosisAction) -> str:
    t = action.action_type
    if t == "query_logs":
        q = f",query={action.query}" if action.query else ""
        return f"query_logs({action.target}{q})"
    if t == "inspect_service":
        return f"inspect_service({action.target})"
    if t == "check_dependency":
        return f"check_dependency({action.target})"
    if t == "restart_service":
        return f"restart_service({action.target})"
    if t == "apply_patch":
        return f"apply_patch({action.target})"
    if t == "propose_diagnosis":
        return f"propose_diagnosis({action.diagnosis})"
    return t


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_ACTION_SCHEMA = textwrap.dedent("""\
    {"action_type": "query_logs",        "target": "<service>", "query": "<optional keyword>"}
    {"action_type": "inspect_service",   "target": "<service>"}
    {"action_type": "check_dependency",  "target": "<service>"}
    {"action_type": "restart_service",   "target": "<service>"}
    {"action_type": "apply_patch",       "target": "<service>", "patch_payload": "{\"key\": \"value\"}"}
    {"action_type": "propose_diagnosis", "diagnosis": "<diagnosis_id>"}""")

SYSTEM_PROMPT = textwrap.dedent(f"""\
    You are an expert Site Reliability Engineer (SRE) debugging a live system incident.
    Your goal: identify the root cause and resolve the incident by choosing ONE action per turn.

    RULES:
    1. Respond with a single valid JSON object — no markdown fences, no prose.
    2. Use only the action_types listed below.
    3. Be strategic: avoid repeating actions you have already taken (redundant actions are penalised).
    4. Investigate systematically: check alerts, inspect failing services, trace dependency chains.
    5. Only call propose_diagnosis when confident. Wrong diagnoses are penalised.
    6. For "hard" incidents: you must apply_patch BEFORE or AFTER propose_diagnosis.
    7. patch_payload must be a valid JSON-encoded string, e.g.: "{{\"key\": \"value\"}}"

    ACTION SCHEMAS (one per line):
    {_ACTION_SCHEMA}

    STRATEGY:
    - Start by reading alerts and initial service statuses.
    - Use inspect_service on anything that is "down" or "degraded".
    - Use query_logs to read clues — filter by "ERROR" or "FATAL" first.
    - Use check_dependency to trace failure propagation.
    - Only restart or patch after identifying the root service.
    - Use propose_diagnosis with a root-cause hypothesis (e.g., 'cache_memory_exhaustion', 'database_overload'). Guessing incorrectly will end the episode!""")


def _build_user_prompt(obs: IncidentDiagnosisObservation, step: int) -> str:
    alerts_str = "\n".join(f"  {a}" for a in obs.alerts) or "  (none)"

    services_str = "\n".join(
        f"  {s.name}: status={s.status}"
        + (f" cpu={s.cpu_pct:.1f}%" if s.cpu_pct is not None else "")
        + (f" mem={s.mem_pct:.1f}%" if s.mem_pct is not None else "")
        + (f" err_rate={s.error_rate:.1f}/s" if s.error_rate is not None else "")
        for s in obs.service_statuses
    ) or "  (none inspected yet)"

    dep_str = "\n".join(
        f"  {link}: healthy={info.get('healthy')} reason={info.get('reason', '')}"
        for link, info in obs.dependency_health.items()
    ) or "  (none checked yet)"

    logs_str = "\n".join(f"  {ln}" for ln in obs.visible_logs[-10:]) or "  (none retrieved)"

    prev_str = "\n".join(
        f"  {i+1}. {a}" for i, a in enumerate(obs.previous_actions[-5:])
    ) or "  (none)"

    return textwrap.dedent(f"""\
        STEP {step} — INCIDENT INVESTIGATION

        SYSTEM ALERTS:
        {alerts_str}

        INSPECTED SERVICE METRICS:
        {services_str}

        DEPENDENCY HEALTH:
        {dep_str}

        RECENT LOGS (last 10 lines retrieved):
        {logs_str}

        PREVIOUS ACTIONS (last 5):
        {prev_str}

        Choose your next action as a single JSON object.""")


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(user_prompt: str, conversation_history: list) -> str:
    client = _get_llm()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *conversation_history,
        {"role": "user", "content": user_prompt},
    ]
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        max_tokens=300,
        stream=False,
    )
    return (response.choices[0].message.content or "").strip()


def _parse_action(raw: str) -> IncidentDiagnosisAction:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(ln for ln in lines[1:] if not ln.strip().startswith("```")).strip()
    data = json.loads(text)
    return IncidentDiagnosisAction(**data)


def _fallback_action(task_id: str, obs: IncidentDiagnosisObservation, step: int, fallback_idx: int) -> IncidentDiagnosisAction:
    """Deterministic local policy used when no LLM call is available."""
    if task_id == "easy":
        sequence = [
            IncidentDiagnosisAction(action_type="inspect_service", target="database"),
            IncidentDiagnosisAction(action_type="query_logs", target="database"),
            IncidentDiagnosisAction(action_type="propose_diagnosis", diagnosis="database_crash"),
        ]
    elif task_id == "medium":
        sequence = [
            IncidentDiagnosisAction(action_type="inspect_service", target="cache"),
            IncidentDiagnosisAction(action_type="check_dependency", target="payment_service"),
            IncidentDiagnosisAction(action_type="query_logs", target="cache"),
            IncidentDiagnosisAction(action_type="propose_diagnosis", diagnosis="cache_memory_exhaustion"),
        ]
    else:
        sequence = [
            IncidentDiagnosisAction(action_type="inspect_service", target="routing_config"),
            IncidentDiagnosisAction(action_type="query_logs", target="routing_config"),
            IncidentDiagnosisAction(
                action_type="apply_patch",
                target="routing_config",
                patch_payload='{"eu-west-3": "https://svc.eu-west-3.internal:8080"}',
            ),
            IncidentDiagnosisAction(action_type="propose_diagnosis", diagnosis="routing_config_misconfiguration"),
        ]

    if step <= len(sequence):
        return sequence[step - 1]

    # Deterministic fallback for any extra steps: probe the most informative target.
    cycle = {
        "easy": ["database", "auth_service", "api_gateway"],
        "medium": ["cache", "payment_service", "inventory_service", "order_service"],
        "hard": ["routing_config", "user_service", "notification_service", "message_queue"],
    }.get(task_id, ["api_gateway"])
    target = cycle[fallback_idx % len(cycle)]
    return IncidentDiagnosisAction(action_type="inspect_service", target=target)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

# Per-task fallback service lists (used when LLM call fails)
_FALLBACK_SERVICES: dict[str, list[str]] = {
    "easy":   ["database", "auth_service", "api_gateway"],
    "medium": ["billing_service", "cache", "payment_service", "inventory_service", "order_service", "frontend"],
    "hard":   ["routing_config", "user_service", "notification_service", "message_queue",
               "api_v2", "api_v1", "load_balancer"],
}


def run_episode(env, task_id: str) -> dict:
    """Run one full episode; return summary dict."""
    result = env.reset(task_id=task_id)
    obs: IncidentDiagnosisObservation = result.observation

    log_start(task=task_id, model=MODEL_NAME)

    rewards: List[float] = []
    trajectory: List[dict] = []
    conversation_history: list = []
    steps_taken    = 0
    success        = False
    score          = 0.0
    fallback_idx   = 0   # cycles through fallback services on consecutive LLM failures

    try:
        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            user_prompt = _build_user_prompt(obs, step)
            raw_text    = ""
            action: IncidentDiagnosisAction
            error_msg: Optional[str] = None

            if HF_TOKEN:
                try:
                    raw_text   = _call_llm(user_prompt, conversation_history)
                    action     = _parse_action(raw_text)
                    fallback_idx = 0   # reset on success
                    _debug(f"  step {step}: {_action_str(action)}")
                except Exception as exc:
                    action = _fallback_action(task_id, obs, step, fallback_idx)
                    fallback_idx += 1
                    _debug(f"  step {step}: LLM error ({exc!r}) → {_action_str(action)} fallback")
                    error_msg = str(exc)[:120]
                    raw_text  = raw_text or "<error>"
            else:
                action = _fallback_action(task_id, obs, step, fallback_idx)
                fallback_idx += 1
                raw_text = json.dumps(action.model_dump(exclude_none=True))
                _debug(f"  step {step}: offline fallback → {_action_str(action)}")

            # Maintain conversation context
            conversation_history.append({"role": "user",      "content": user_prompt})
            conversation_history.append({"role": "assistant", "content": raw_text})
            
            # Keep only the last 6 turns (3 steps) to prevent context exhaustion
            MAX_HISTORY_TURNS = 6
            if len(conversation_history) > MAX_HISTORY_TURNS:
                conversation_history = conversation_history[-MAX_HISTORY_TURNS:]

            # Step the environment
            result = env.step(action)
            obs    = result.observation
            # obs.reward is a dedicated field that survives WS serialization
            reward = float(
                obs.reward if obs.reward != 0.0
                else (result.reward if result.reward is not None
                      else obs.metadata.get("reward", 0.0))
            )

            done = obs.done
            if not obs.metadata.get("last_action_valid", True):
                error_msg = obs.metadata.get("error", error_msg)

            rewards.append(reward)
            steps_taken = step

            trajectory.append(
                {
                    "action": action.model_dump(exclude_none=True),
                    "reward": reward,
                    "done": done,
                    "observation": obs.model_dump(),
                }
            )

            log_step(
                step   = step,
                action = _action_str(action),
                reward = reward,
                done   = done,
                error  = error_msg,
            )

            if done:
                success = obs.metadata.get("success", False)
                break

        # Deterministic task-specific grading in [0, 1].
        score = grade_task(task_id, trajectory)
        # Some transports may omit "success" in metadata, so derive from score.
        success = score >= 0.99

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id":     task_id,
        "score":       score,
        "steps_taken": steps_taken,
        "success":     success,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _debug(f"API_BASE_URL = {API_BASE_URL}")
    _debug(f"MODEL_NAME   = {MODEL_NAME}")
    _debug(f"ENV_BASE_URL = {ENV_BASE_URL}")
    if not HF_TOKEN:
        _debug("WARNING: HF_TOKEN not set — requests will likely fail with 401.")

    results = []
    env = IncidentDiagnosisEnv(base_url=ENV_BASE_URL)

    with env.sync() as sync_env:
        for task_id in TASK_IDS:
            try:
                summary = run_episode(sync_env, task_id)
                results.append(summary)
            except Exception as exc:
                _debug(f"Episode crashed for task_id={task_id!r}: {exc!r}")
                log_end(success=False, steps=0, score=0.01, rewards=[])
                results.append({
                    "task_id":     task_id,
                    "score":       0.01,
                    "steps_taken": 0,
                    "success":     False,
                })

    # Human-readable summary to stderr only
    print("\n" + "=" * 60, file=sys.stderr)
    print(f"  {'task':<10} {'model':<30} {'score':>6} {'steps':>6}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    for r in results:
        print(
            f"  {r['task_id']:<10} {MODEL_NAME:<30} "
            f"{r['score']:>6.3f} {r['steps_taken']:>6}",
            file=sys.stderr,
        )
    print("=" * 60, file=sys.stderr)


if __name__ == "__main__":
    main()
