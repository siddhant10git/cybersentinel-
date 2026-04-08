"""
CyberSentinel — Async environment client.

Provides the CyberSentinelEnv class with:
  - from_docker_image(image_name)  — launch container and connect
  - reset()  → StepResult
  - step(action)  → StepResult
  - state()  → dict
  - close()  — stop and remove container
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from models import Action, Classification


# ── Result types ─────────────────────────────────────────────────────────────

@dataclass
class CyberSentinelObservation:
    """Observation returned by the environment."""
    pending_alerts: list[dict]
    intel_reports: list[dict]
    current_alert: Optional[dict]
    time_remaining: int
    alerts_processed: int
    alerts_total: int
    current_score: float


@dataclass
class CyberSentinelStepResult:
    """Result of reset() or step()."""
    observation: CyberSentinelObservation
    reward: float = 0.0
    done: bool = False
    info: dict = field(default_factory=dict)


def _parse_observation(data: dict) -> CyberSentinelObservation:
    return CyberSentinelObservation(
        pending_alerts=data.get("pending_alerts", []),
        intel_reports=data.get("intel_reports", []),
        current_alert=data.get("current_alert"),
        time_remaining=data.get("time_remaining", 0),
        alerts_processed=data.get("alerts_processed", 0),
        alerts_total=data.get("alerts_total", 0),
        current_score=data.get("current_score", 0.0),
    )


# ── Environment client ──────────────────────────────────────────────────────

class CyberSentinelEnv:
    """Async client for the CyberSentinel environment."""

    def __init__(self, base_url: str, container_id: Optional[str] = None):
        self._base_url = base_url.rstrip("/")
        self._container_id = container_id
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=30)

    @classmethod
    async def from_docker_image(
        cls,
        image_name: str,
        port: Optional[int] = None,
        timeout: int = 30,
    ) -> "CyberSentinelEnv":
        """Launch a Docker container from the given image and return a connected client."""
        if port is None:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                port = s.getsockname()[1]

        container_id = subprocess.check_output(
            [
                "docker", "run", "-d", "--rm",
                "-p", f"{port}:7860",
                image_name,
            ],
            text=True,
        ).strip()

        base_url = f"http://localhost:{port}"
        env = cls(base_url=base_url, container_id=container_id)

        # Wait for the server to be ready
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = await env._client.get("/health")
                if resp.status_code == 200:
                    return env
            except Exception:
                pass
            await asyncio.sleep(0.5)

        raise TimeoutError(
            f"CyberSentinel container {container_id[:12]} did not become healthy "
            f"within {timeout}s"
        )

    async def reset(self, task_id: str = "easy") -> CyberSentinelStepResult:
        """Reset the environment with the given task."""
        resp = await self._client.post("/reset", json={"task_id": task_id})
        resp.raise_for_status()
        data = resp.json()
        return CyberSentinelStepResult(
            observation=_parse_observation(data),
            reward=0.0,
            done=False,
            info={"task_id": task_id},
        )

    async def step(self, action: Action) -> CyberSentinelStepResult:
        """Submit an action and receive the result."""
        resp = await self._client.post("/step", json=action.model_dump(mode="json"))
        resp.raise_for_status()
        data = resp.json()
        return CyberSentinelStepResult(
            observation=_parse_observation(data.get("observation", data)),
            reward=data.get("reward", 0.0),
            done=data.get("done", False),
            info=data.get("info", {}),
        )

    async def state(self) -> dict:
        """Return the full internal state."""
        resp = await self._client.get("/state")
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        """Stop the Docker container and clean up."""
        await self._client.aclose()
        if self._container_id:
            try:
                subprocess.run(
                    ["docker", "stop", self._container_id],
                    capture_output=True,
                    timeout=15,
                )
            except Exception:
                pass
