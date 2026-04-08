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

> **An OpenEnv environment for training and evaluating AI agents on real-world SRE incident response.**

Agents must **investigate, reason under uncertainty, and fix live production incidents** — not solvable by a single LLM prompt. Requires sequential, strategic exploration of a partially-observable system graph.

---

## 📋 Table of Contents

1. [What is this?](#-what-is-this)
2. [Project Structure](#️-project-structure)
3. [Observation Space](#️-observation-space)
4. [Action Space](#-action-space)
5. [Reward Design](#-reward-design)
6. [Tasks](#-tasks)
7. [Graders](#-graders)
8. [Setup & Running](#-setup--running)
9. [API Endpoints](#-api-endpoints)
10. [Constraints](#️-constraints)
11. [Baseline Scores](#-baseline-scores)

---

## 🎯 What is this?

A **POMDP** (Partially Observable Markov Decision Process) environment where an AI agent acts as an SRE diagnosing production microservice incidents.

| Property | Detail |
|---|---|
| **Domain** | Site Reliability Engineering — incident diagnosis & remediation |
| **Interface** | OpenEnv (`step` / `reset` / `state`) over HTTP + WebSocket |
| **Observability** | Partial — agent builds picture incrementally through investigation |
| **Tasks** | 3 levels: Easy → Medium → Hard |
| **Graders** | Deterministic, reproducible scores in `(0.01, 0.99)` |
| **Deploy** | Hugging Face Space (`yeet00/IncidentDiagnosisEnv`) |

**Real-world motivation:** Production incidents are never transparent. Operators must correlate noisy logs, cascading alerts, and partial metrics under time pressure — this environment reproduces those constraints faithfully.

---

## 🗂️ Project Structure

```
.
├── server/
│   ├── app.py                            # FastAPI server (OpenEnv factory)
│   ├── incident_diagnosis_environment.py # Core POMDP environment logic
│   ├── requirements.txt                  # Server Python dependencies
│   └── __init__.py
│
├── tasks/
│   ├── task_easy.json                    # Single service crash
│   ├── task_medium.json                  # Dependency timeout cascade
│   └── task_hard.json                    # Hidden cascading failure + patch required
│
├── models.py                             # Pydantic Action / Observation / Reward models
├── client.py                             # Typed Python client wrapper
├── graders.py                            # Deterministic per-task grader functions
├── inference.py                          # Baseline agent (OpenAI client + HF Router)
│
├── validate_graders.py                   # Grader determinism & range checks
├── validate_tasks.py                     # Task JSON schema validation
├── validate-submission.sh                # End-to-end pre-submission validator
├── run_test.sh                           # Quick local smoke test
│
├── openenv.yaml                          # OpenEnv spec metadata
├── pyproject.toml                        # Project dependencies (uv)
├── uv.lock                               # Locked dependency tree
├── __init__.py                           # Package init (exports client + models)
└── Dockerfile                            # Production container
```

---

## 👁️ Observation Space

The agent receives a **partial observation** — the full hidden system state is never exposed.

| Field | Type | Description |
|---|---|---|
| `alerts` | `List[str]` | System-level alerts always visible (like a pager notification) |
| `visible_logs` | `List[str]` | Log lines retrieved so far via `query_logs` actions |
| `service_statuses` | `List[ServiceStatus]` | Metrics for services explicitly inspected this episode |
| `dependency_health` | `Dict[str, Any]` | Health of dependency links revealed by `check_dependency` |
| `previous_actions` | `List[str]` | History of all actions taken this episode |
| `step_count` | `int` | Steps elapsed in this episode |
| `done` | `bool` | Whether the episode has terminated |
| `reward` | `float` | Reward from the last action (0.0 on reset) |
| `metadata` | `Dict[str, Any]` | Diagnostic payload: messages, errors, patch status |

**`ServiceStatus` fields** (revealed only after `inspect_service`):

| Sub-field | Type | Description |
|---|---|---|
| `name` | `str` | Service identifier |
| `status` | `"healthy" \| "degraded" \| "down" \| "unknown"` | Runtime status |
| `cpu_pct` | `float?` | CPU utilisation % |
| `mem_pct` | `float?` | Memory utilisation % |
| `error_rate` | `float?` | Errors per second |

**POMDP properties:**

- Logs revealed **3 lines at a time** per `query_logs` call
- Service metrics only visible **after** `inspect_service`
- Dependency health only visible **after** `check_dependency`
- Noise lines injected into logs based on hidden failure mode

---

## 🎮 Action Space

```python
class IncidentDiagnosisAction(Action):
    action_type: Literal[
        "query_logs",        # retrieve log lines from a service
        "inspect_service",   # reveal cpu/mem/error_rate metrics
        "check_dependency",  # reveal outgoing dependency link health
        "restart_service",   # attempt to restart a service (mutates state)
        "apply_patch",       # submit a JSON config patch (required for hard)
        "propose_diagnosis", # end episode with root-cause hypothesis
    ]
    target:        Optional[str]  # service name (required for most actions)
    query:         Optional[str]  # keyword filter for query_logs
    patch_payload: Optional[str]  # JSON-encoded config dict for apply_patch
    diagnosis:     Optional[str]  # root-cause ID for propose_diagnosis
```

**Action effects:**

| Action | Required Fields | Effect |
|---|---|---|
| `query_logs` | `target` | Reveals up to 3 log lines; can filter by keyword |
| `inspect_service` | `target` | Reveals cpu / mem / error_rate for target |
| `check_dependency` | `target` | Reveals health of **outgoing** dependency links from target |
| `restart_service` | `target` | Heals target if it's a valid restart-heal service; else −0.20 |
| `apply_patch` | `target`, `patch_payload` | JSON config; correct keys heal target and dependents |
| `propose_diagnosis` | `diagnosis` | Ends episode; **any wrong answer terminates with penalty** |

---

## 🎯 Reward Design

Rewards provide **dense signal throughout the trajectory** — not just at episode end.

### Step rewards (per action)

| Event | Reward | Notes |
|---|---|---|
| Base step cost | **−0.02** | Downtime clock — incentivises efficiency |
| Redundant action | **−0.05** | Re-querying same logs / re-inspecting same service |
| Useful log revealed | **+0.05** | Non-noise log lines from `query_logs` |
| Unhealthy service inspected | **+0.10** | `status = degraded / down` after `inspect_service` |
| Unhealthy dependency revealed | **+0.08** per link | Via `check_dependency` |
| Correct restart (easy) | **+0.99**† | Episode ends, system recovered |
| Correct restart (other tasks) | **+0.30** | Partial heal — root cause may remain |
| Wrong restart | **−0.20** | Service not in root-cause chain |
| Correct patch applied | **+0.50** | Partial credit; must still `propose_diagnosis` |
| Wrong patch | **−0.10** | Missing or incorrect config keys |
| Wrong diagnosis | **−0.30** | Episode immediately ends |
| Timeout (max steps exceeded) | **−0.10** | Applied at step limit |

> †Grader score, not raw step reward — see Graders section.

### Episode (grader) scores

| Outcome | Score |
|---|---|
| Correct diagnosis (easy / medium) | **0.99** |
| Correct diagnosis **+** correct patch (hard) | **0.99** |
| Correct diagnosis, missing patch (hard) | **0.50** |
| Strong investigation, no resolution | **0.07 – 0.27** |
| No meaningful investigation | **0.01** |

**Anti-hacking principle:** Investigative rewards are capped — the trajectory budget is dominated by the final resolution reward, preventing reward-farming without solving the task.

---

## 🧪 Tasks

### 🟢 Easy — Single Service Crash
```
Max steps: 15  |  Difficulty: Low  |  Root cause: database_crash
```
- **Scenario:** Database crashes → auth_service retry loop → api_gateway 503s
- **Signals:** Clear OOM logs on `database`, cascading error rates visible after inspection
- **Resolution:** `restart_service(database)` **OR** `propose_diagnosis(database_crash)`
- **Grader:** Full credit for correct resolution; partial for useful investigation

### 🟡 Medium — Dependency Timeout Cascade
```
Max steps: 20  |  Difficulty: Medium  |  Root cause: cache_memory_exhaustion
```
- **Scenario:** Cache memory exhaustion causes all downstream services to timeout
- **Deception:** `order_service` logs blame `payment_service` / `inventory_service`, not cache
- **Resolution:** Must trace dependency graph back to `cache` → `propose_diagnosis(cache_memory_exhaustion)`
- **Grader:** Full credit for correct diagnosis; partial for seeing cache-related deps

### 🔴 Hard — Hidden Cascading Failure (Patch Required)
```
Max steps: 25  |  Difficulty: High  |  Root cause: routing_config_misconfiguration
```
- **Scenario:** Routing config injected 2h ago sends EU traffic to wrong endpoints; `load_balancer` appears healthy, routing health check returns 200 but the config is wrong
- **Deception:** Multiple services show errors but root cause is a config key mismatch
- **Resolution:** `apply_patch(routing_config, {"eu-west-3": "..."})` **+** `propose_diagnosis(routing_config_misconfiguration)`
- **Grader:** Full credit only for correct diagnosis + correct patch; 0.50 for diagnosis alone

---

## 🧮 Graders

Each task has its own deterministic grader in `graders.py`:

```python
grade_easy(trajectory)   -> float  # strictly in (0.01, 0.99)
grade_medium(trajectory) -> float  # strictly in (0.01, 0.99)
grade_hard(trajectory)   -> float  # strictly in (0.01, 0.99)
grade_task(task_id, trajectory)    # dispatcher — use this in inference.py
```

**Grader contract:**

| Guarantee | Detail |
|---|---|
| Deterministic | Same trajectory always produces the same score |
| Range | Strictly `(0.01, 0.99)` — never exactly 0 or 1 |
| Partial credit | Non-zero scores for meaningful investigation progress |
| Linked to tasks | Each task JSON references its grader: `"grader": "graders:grade_easy"` |

---

## 🚀 Setup & Running

### Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | 3.10+ | [python.org](https://python.org) |
| uv | latest | `pip install uv` |
| openenv CLI | latest | `pip install openenv-core` |
| Docker | any stable | [docker.com](https://docker.com) |

### Install dependencies

```bash
uv sync
```

### Run the server locally

```bash
# Option A — uvicorn directly
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Option B — uv project script
uv run --project . server
```

### Run the baseline agent

```bash
export HF_TOKEN=hf_...                              # or OPENAI_API_KEY=sk_...
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ENV_BASE_URL=http://localhost:8000

python inference.py
```

**Inference env vars:**

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | ✅ | — | HF / OpenAI API key (also accepts `OPENAI_API_KEY`) |
| `API_BASE_URL` | ✅ | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | ✅ | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `ENV_BASE_URL` | ✅ | `http://localhost:8000` | Running environment server |
| `LOCAL_IMAGE_NAME` | ❌ | — | Docker image name (if using `from_docker_image()`) |
| `MAX_STEPS` | ❌ | `40` | Override per-run step cap |

### Run with Docker

```bash
docker build -t incident-diagnosis-env .
docker run -p 8000:8000 -e HF_TOKEN=$HF_TOKEN incident-diagnosis-env
```

### Validate everything

```bash
# Grader determinism + range checks
python validate_graders.py

# Task JSON schema checks
python validate_tasks.py

# OpenEnv spec compliance
uv run openenv validate

# Full pre-submission end-to-end check
./validate-submission.sh https://<your-space>.hf.space .
```

### Deploy to Hugging Face Spaces

```bash
export HUGGING_FACE_HUB_TOKEN=hf_...
uv run openenv push
```

Space includes:
- **Web UI** at `/web` — interactive task explorer
- **WebSocket** at `/ws` — persistent eval session

---

## 🔌 API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start new episode; body: `{"task_id": "easy"}` |
| `POST` | `/step` | Submit action; body: action JSON object |
| `GET` | `/state` | Current episode state (episode_id, step_count) |
| `GET` | `/tasks` | List available task IDs: `["easy", "medium", "hard"]` |
| `GET` | `/schema` | Action + Observation JSON schemas |
| `GET` | `/health` | Health check |
| `WS` | `/ws` | Persistent WebSocket session |

**Quick test:**

```bash
# 1. Reset to easy task
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'

# 2. Inspect database service
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "inspect_service", "target": "database"}'

# 3. Propose diagnosis
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "propose_diagnosis", "diagnosis": "database_crash"}'
```

---

## ⚙️ Constraints

| Constraint | Value |
|---|---|
| Max inference runtime | < 20 minutes (all 3 tasks) |
| Memory | 8 GB RAM |
| CPU | 2 vCPU |
| Per-task step limits | Easy: 15 · Medium: 20 · Hard: 25 |
| Grader score range | Strictly `(0.01, 0.99)` — never exactly 0 or 1 |
| Reproducibility | Deterministic hidden state — same `task_id` = same scenario |
| State backend | Pure Python in-memory — no real services, no external calls |

---

## 📊 Baseline Scores

**Run configuration:** `inference.py` · `temperature=0.0` · tasks in order `easy → medium → hard`

### Offline / fallback baseline (no LLM — always reproducible)

When `HF_TOKEN` is not set, `inference.py` uses a built-in deterministic policy:

| Task | Score | Policy |
|---|---|---|
| easy | **0.99** | `inspect_service(database)` → `query_logs(database)` → `propose_diagnosis(database_crash)` |
| medium | **0.99** | `inspect_service(cache)` → `check_dependency(payment_service)` → `query_logs(cache)` → `propose_diagnosis(cache_memory_exhaustion)` |
| hard | **0.99** | `inspect_service(routing_config)` → `query_logs(routing_config)` → `apply_patch(routing_config, {...})` → `propose_diagnosis(routing_config_misconfiguration)` |

### LLM agent baseline (Qwen2.5-72B-Instruct · `temperature=0.0` · expected range)

| Task | Score Range | Notes |
|---|---|---|
| easy | 0.70 – 0.99 | Clear signals; most frontier models solve reliably |
| medium | 0.40 – 0.85 | Misleading logs require multi-step deduction |
| hard | 0.10 – 0.50 | Requires precise patch payload + diagnosis; deceptive signals |

> All scores strictly in open interval `(0.0, 1.0)` — never exactly 0 or 1.
