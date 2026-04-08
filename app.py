"""
CyberSentinel — FastAPI server exposing the OpenEnv interface.

Endpoints:
  POST /reset   — start/restart a task
  POST /step    — submit an action
  GET  /state   — current environment state
  GET  /tasks   — list available tasks
  GET  /health  — liveness probe
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from environment import CyberSentinelEnv
from models import Action, EnvState, Observation, StepResult
from tasks import load_tasks

app = FastAPI(
    title="CyberSentinel",
    description="OpenEnv threat-intelligence triage environment",
    version="1.0.0",
)

# Single environment instance (stateful)
env = CyberSentinelEnv()


# ── Request / Response schemas ───────────────────────────────────────────────

from typing import Optional

class ResetRequest(BaseModel):
    task_id: str = "easy"


class TaskInfo(BaseModel):
    task_id: str
    name: str
    difficulty: str
    description: str
    num_alerts: int
    max_steps: int


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
    """Return the full internal state."""
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


@app.get("/health")
def health():
    """Liveness probe."""
    return {"status": "ok", "environment": "cybersentinel", "version": "1.0.0"}


@app.get("/")
def read_root():
    """Root endpoint for Hugging Face Spaces health checks and browser visits."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")
