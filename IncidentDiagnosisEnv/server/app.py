"""
FastAPI application for the Incident Diagnosis Environment.

Auto-generated endpoints (via OpenEnv create_app factory):
    POST /reset   — Reset the environment (accepts task_id in body)
    POST /step    — Execute an action
    GET  /state   — Current episode state
    GET  /schema  — Action / observation JSON schemas
    WS   /ws      — Persistent WebSocket session

Custom routes:
    GET  /tasks   — List available task IDs

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from typing import List

try:
    from openenv import create_app
except ImportError:
    from openenv.core.env_server.http_server import create_app  # type: ignore[no-redef]

try:
    from ..models import IncidentDiagnosisAction, IncidentDiagnosisObservation
    from .incident_diagnosis_environment import IncidentDiagnosisEnvironment
except ImportError:
    from models import IncidentDiagnosisAction, IncidentDiagnosisObservation  # type: ignore[no-redef]
    from server.incident_diagnosis_environment import IncidentDiagnosisEnvironment  # type: ignore[no-redef]


# ── Create the app via the OpenEnv factory ─────────────────────────────────
app = create_app(
    IncidentDiagnosisEnvironment,
    IncidentDiagnosisAction,
    IncidentDiagnosisObservation,
    env_name="IncidentDiagnosisEnv",
    max_concurrent_envs=1,
)


# ── GET /tasks ─────────────────────────────────────────────────────────────
@app.get(
    "/tasks",
    summary="List available task IDs",
    response_model=List[str],
    tags=["tasks"],
)
async def list_tasks() -> List[str]:
    """Return the list of task IDs that can be passed to ``/reset``.

    * ``"easy"``   — Single service crash. Clear logs. Identify root cause.
    * ``"medium"`` — Dependency timeout cascade. Misleading logs. Trace failure.
    * ``"hard"``   — Hidden cascading failure. Requires correct patch + diagnosis.
    """
    return ["easy", "medium", "hard"]


# ── Entry point ────────────────────────────────────────────────────────────
def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Incident Diagnosis Env server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
