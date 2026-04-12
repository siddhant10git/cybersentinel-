"""
CyberSentinel — FastAPI server exposing the OpenEnv interface (v2).

Endpoints:
  POST /reset    — start/restart a task
  POST /step     — submit an action
  GET  /state    — current environment state
  GET  /tasks    — list available tasks
  POST /grade    — grade a completed episode from a list of actions
  GET  /health   — liveness probe
"""

from __future__ import annotations

import socket
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from environment import CyberSentinelEnv
from graders import grade_task, DEFAULT_CONFIG
from models import Action, EnvState, Observation, StepResult
from tasks import load_tasks

app = FastAPI(
    title="CyberSentinel",
    description=(
        "OpenEnv threat-intelligence triage environment. "
        "Agents classify security alerts, assign priorities, and resist adversarial disinformation. "
        "5 tasks: easy, medium, hard, insider, zero."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow cross-origin requests (useful for Hugging Face Spaces preview)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single environment instance (stateful per-session)
env = CyberSentinelEnv()


# ── Request / Response schemas ───────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "easy"


class TaskInfo(BaseModel):
    task_id: str
    name: str
    difficulty: str
    description: str
    num_alerts: int
    max_steps: int


class GradeRequest(BaseModel):
    task_id: str
    actions: list[dict[str, Any]]


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/reset", response_model=Observation)
def reset(req: Optional[ResetRequest] = None):
    """Load a task and return the initial observation."""
    task_id = req.task_id if req else "easy"
    try:
        obs = env.reset(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return obs


@app.post("/step", response_model=StepResult)
def step(action: Action):
    """Submit an action and receive (observation, reward, done, info)."""
    try:
        result = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@app.get("/state", response_model=EnvState)
def state():
    """Return the full internal state snapshot."""
    try:
        return env.state()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/tasks", response_model=list[TaskInfo])
def list_all_tasks():
    """List all available tasks."""
    tasks = load_tasks()
    return [
        TaskInfo(
            task_id=t["task_id"],
            name=t["name"],
            difficulty=t["difficulty"],
            description=t["description"],
            num_alerts=len(t["alerts"]),
            max_steps=t["max_steps"],
        )
        for t in tasks.values()
    ]


@app.post("/grade")
def grade_episode(req: GradeRequest):
    """
    Grade a completed episode from a list of actions.

    Accepts raw action dicts (same schema as /step body).
    Returns the full GradeResult including per-alert breakdown.
    """
    tasks = load_tasks()
    if req.task_id not in tasks:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {req.task_id!r}")

    task = tasks[req.task_id]
    try:
        actions = [Action(**a) for a in req.actions]
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Action parse error: {exc}")

    result = grade_task(
        actions=actions,
        alerts=task["alerts"],
        task_id=req.task_id,
        config=DEFAULT_CONFIG,
    )
    return result


@app.get("/health")
def health():
    """Liveness probe."""
    return {
        "status": "ok",
        "environment": "cybersentinel",
        "version": "2.0.0",
        "tasks_available": list(load_tasks().keys()),
    }


@app.get("/")
def read_root():
    """Root endpoint — redirect to API docs."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


def _check_port_available(port: int) -> None:
    """Raise a clear error if the port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
        except OSError:
            raise RuntimeError(
                f"\n❌  Port {port} is already in use.\n"
                f"   Fix: kill the process using that port, or set a different port with\n"
                f"   uvicorn server.app:app --port <other_port>\n"
            )


def main():
    import uvicorn
    _check_port_available(7860)
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
