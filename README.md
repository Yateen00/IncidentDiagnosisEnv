---
title: Incident Diagnosis Env
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# 🧠 Incident Diagnosis Environment

**An interactive, partially-observable system debugging environment for training and evaluating AI agents.**

Built for the OpenEnv hackathon — agents must investigate, reason under uncertainty, and fix live system incidents.

---

## 🎯 What is this?

This is a **POMDP (Partially Observable Markov Decision Process)** environment where an AI agent acts as an SRE (Site Reliability Engineer) diagnosing production incidents.

The agent:
- **Cannot see the full system state** — must investigate incrementally
- **Receives partial, noisy information** via investigative actions
- **Accrues step costs** (downtime clock) — must be efficient
- **Must identify root cause** and (for hard tasks) **apply a correct patch**

This is **not solvable by a single LLM prompt** — it requires sequential, strategic reasoning.

---

## 🗂️ Project Structure

```
IncidentDiagnosisEnv/
├── server/
│   ├── app.py                              # FastAPI server (OpenEnv factory)
│   └── incident_diagnosis_environment.py   # Core environment logic
├── tasks/
│   ├── task_easy.json                      # Single service crash
│   ├── task_medium.json                    # Dependency timeout cascade
│   └── task_hard.json                      # Hidden cascading failure + patch required
├── models.py                               # Pydantic observation/action types
├── client.py                               # Typed EnvClient wrapper
├── inference.py                            # Baseline agent (HF Router + OpenAI client)
├── openenv.yaml                            # OpenEnv spec
├── pyproject.toml                          # Dependencies
└── Dockerfile                              # Production container
```

---

## 👁️ Observation Space

The agent receives a **partial observation** — never the full hidden system state.

| Field | Type | Description |
|---|---|---|
| `visible_logs` | `List[str]` | Log lines retrieved so far via `query_logs` actions |
| `service_statuses` | `List[ServiceStatus]` | Metrics for services the agent has explicitly inspected |
| `dependency_health` | `Dict[str, Any]` | Health of dependency links revealed by `check_dependency` |
| `alerts` | `List[str]` | High-level system alerts always visible (low-info surface) |
| `previous_actions` | `List[str]` | History of actions taken this episode |
| `step_count` | `int` | Steps taken so far |
| `done` | `bool` | Whether the episode has ended |

### POMDP Properties
- Logs are revealed `3 lines at a time` per `query_logs` call
- Metrics only visible **after** `inspect_service` is called
- Dependency health only visible **after** `check_dependency` is called
- Log noise injected based on hidden failure mode

---

## 🎮 Action Space

```python
class IncidentDiagnosisAction(Action):
    action_type: Literal[
        "query_logs",        # reveals log lines from a service
        "inspect_service",   # reveals full metrics for a service
        "check_dependency",  # reveals health of dependency links
        "restart_service",   # attempts to restart a service (mutates state)
        "apply_patch",       # submits a config patch (required for hard task)
        "propose_diagnosis", # ends episode with root-cause hypothesis
    ]
    target:        Optional[str]   # service to act on
    query:         Optional[str]   # log keyword filter (for query_logs)
    patch_payload: Optional[str]   # JSON-encoded config patch (for apply_patch)
    diagnosis:     Optional[str]   # root-cause ID (for propose_diagnosis)
```

### Action Effects

| Action | Effect |
|---|---|
| `query_logs` | Reveals 3 log lines from target service; can filter by keyword |
| `inspect_service` | Reveals cpu/mem/error_rate for target service |
| `check_dependency` | Reveals health of outgoing dependency links from target |
| `restart_service` | Heals target if it's the root cause service, else -0.20 |
| `apply_patch` | Accepts JSON config; correct keys heal target service |
| `propose_diagnosis` | Ends episode; correct = +1.0, wrong = -0.30. ANY wrong diagnosis ends the episode! |

---

## 🎯 Reward Design

| Event | Reward |
|---|---|
| Base step cost | **-0.02** (downtime clock) |
| Redundant action | **-0.05** (penalises re-querying same info) |
| Useful log revealed | **+0.05** |
| Unhealthy service inspected | **+0.10** |
| Unhealthy dep link revealed | **+0.08** per link |
| Correct restart | **+1.00** (easy), **+0.30** (partial) |
| Wrong restart | **-0.20** |
| Correct patch key accepted | **+0.50** |
| Wrong patch | **-0.10** |
| Correct diagnosis (easy/medium) | **+1.00** |
| Correct diagnosis (hard, patch also correct) | **+1.00** |
| Partial credit (hard: correct diagnosis, no patch) | **+0.20** |
| Wrong diagnosis | **-0.30** |
| Timeout (max steps) | **-0.10** |

**Key design principle:** Information gathering never accumulates unbounded rewards — the total budget is dominated by the final resolution reward. This eliminates reward hacking.

---

## 🧪 Tasks

### 🟢 Easy — Single Service Crash
- **Failure mode:** Database crashes, causing auth_service loop, causing api_gateway 503s
- **Goal:** Identify `database_crash` as root cause
- **Resolution:** `restart_service(database)` **OR** `propose_diagnosis(database_crash)`
- **Max steps:** 15

### 🟡 Medium — Dependency Timeout Cascade
- **Failure mode:** Cache memory exhaustion causes all dependent services to timeout
- **Misleading signal:** order_service logs blame payment_service/inventory_service, not cache
- **Goal:** Trace through to `cache_memory_exhaustion`
- **Resolution:** `propose_diagnosis(cache_memory_exhaustion)` (patch optional for extra credit)
- **Max steps:** 20

### 🔴 Hard — Hidden Cascading Failure (Patch Required)
- **Failure mode:** Routing config misconfiguration injected 2h ago causes EU user failures and notification outage
- **Deceptive signals:** load_balancer shows healthy, routing_config health check returns 200 but config is wrong
- **Goal:** Identify `routing_config_misconfiguration` AND apply correct patch
- **Resolution:** `apply_patch(routing_config, {"eu-west-3": "..."})` + `propose_diagnosis(...)`
- **Max steps:** 25

---

## 🧮 Graders

Each task has a deterministic grader:

```python
def grader(trajectory, final_answer):
    if final_answer == ground_truth and (patch_correct or task != "hard"):
        return 1.0
    elif final_answer == ground_truth:
        return 0.5   # correct diagnosis, wrong/missing patch
    elif diagnosis_attempted and close_to_correct:
        return 0.2   # partial credit
    else:
        return 0.0
```

All scores are in `[0.0, 1.0]`, deterministic, and reproducible.

---

## 🚀 Setup & Running

### Deploying to Hugging Face Spaces

You can deploy the OpenEnv environment directly to Hugging Face Spaces using the `openenv push` command.

```bash
# Export your token
export HUGGING_FACE_HUB_TOKEN=hf_...

# Push the environment
openenv push
```

The deployed Space includes:
- **Web Interface** at `/web`: interactive UI for exploring tasks.
- **WebSocket** at `/ws`: persistent session endpoint for evaluation.

### Prerequisites
- Python 3.10+
- `uv` package manager (or `pip`)
- HF Token with inference access

### Install
```bash
cd IncidentDiagnosisEnv
uv sync
```

### Run server
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
# or:
uv run --project . server
```

### Run baseline agent
```bash
export HF_TOKEN=hf_...
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ENV_BASE_URL=http://localhost:8000

python inference.py
```

### Run with Docker
```bash
docker build -t incident-diagnosis-env .
docker run -p 8000:8000 -e HF_TOKEN=$HF_TOKEN incident-diagnosis-env
```

---

## 🔌 API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Reset environment, returns initial observation |
| `POST` | `/step` | Submit action, returns observation + reward |
| `GET` | `/state` | Current episode state |
| `GET` | `/tasks` | List available task IDs |
| `GET` | `/schema` | Action/observation JSON schemas |
| `WS` | `/ws` | Persistent WebSocket session |

### Example
```bash
# Reset to easy task
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'

# Take a step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "inspect_service", "target": "database"}'
```

---

## ⚙️ Constraints

- Runtime: < 20 minutes per full evaluation
- Resources: 2 vCPU, 8 GB RAM (synthetic state machine — no real services)
- All hidden state is pure Python in-memory
- Deterministic: same `task_id` always produces same hidden scenario
- Reproducible: no randomness in task data

---

## 🧠 Why This Environment Is Interesting

| Property | Status |
|---|---|
| Partial observability | ✅ Agent cannot see full system state |
| Sequential reasoning | ✅ Must probe incrementally over multiple steps |
| Real-world relevance | ✅ Mimics SRE incident response workflow |
| Reward hacking resistant | ✅ Step costs + capped investigative rewards |
| Deterministic grading | ✅ No randomness in scoring |
| Not LLM-solvable in one shot | ✅ Requires multi-turn investigation |
| Increasing difficulty | ✅ Easy → Medium → Hard with different failure modes |
