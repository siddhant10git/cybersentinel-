"""
CyberSentinel — Environment core.

Implements the OpenEnv interface: reset() / step() / state().
"""

from __future__ import annotations

import copy
from typing import Any

from models import (
    Action,
    Alert,
    Classification,
    EnvState,
    Observation,
    Reward,
    StepResult,
    ThreatIntelReport,
)
from tasks import get_task


class CyberSentinelEnv:
    """Threat-intelligence triage environment."""

    def __init__(self) -> None:
        self._task_id: str = ""
        self._task_meta: dict = {}
        self._alerts: list[Alert] = []
        self._intel_reports: list[ThreatIntelReport] = []
        self._pending_ids: list[str] = []
        self._current_idx: int = 0
        self._step_count: int = 0
        self._max_steps: int = 0
        self._done: bool = True
        self._scores: dict[str, float] = {}
        self._reward_details: dict[str, dict] = {}

    # ── helpers ──────────────────────────────────────────────────────────

    def _alert_by_id(self, alert_id: str) -> Alert | None:
        for a in self._alerts:
            if a.id == alert_id:
                return a
        return None

    def _safe_obs_alert(self, alert: Alert) -> dict[str, Any]:
        """Return alert dict with ground-truth fields stripped."""
        d = alert.model_dump(mode="json")
        d.pop("ground_truth_classification", None)
        d.pop("ground_truth_priority", None)
        d.pop("relevant_iocs", None)
        return d

    def _safe_obs_intel(self, report: ThreatIntelReport) -> dict[str, Any]:
        """Return intel dict with hidden fields stripped."""
        d = report.model_dump(mode="json")
        d.pop("is_disinformation", None)
        return d

    def _build_observation(self) -> Observation:
        pending = [
            self._safe_obs_alert(a) for a in self._alerts
            if a.id in self._pending_ids
        ]
        current = None
        if self._current_idx < len(self._alerts) and not self._done:
            current = self._safe_obs_alert(self._alerts[self._current_idx])

        processed = len(self._scores)
        avg = sum(self._scores.values()) / max(processed, 1)

        return Observation(
            pending_alerts=pending,
            intel_reports=[self._safe_obs_intel(r) for r in self._intel_reports],
            current_alert=current,
            time_remaining=max(self._max_steps - self._step_count, 0),
            alerts_processed=processed,
            alerts_total=len(self._alerts),
            current_score=round(avg, 4),
        )

    # ── reward computation ───────────────────────────────────────────────

    def _compute_reward(self, action: Action, alert: Alert) -> Reward:
        """Compute per-step reward with partial-credit signals."""
        classification_score = 0.0
        priority_score = 0.0
        justification_score = 0.0
        disinformation_bonus = 0.0
        time_penalty = 0.0

        # 1. Classification accuracy (0.0 or 0.35)
        if action.classification == alert.ground_truth_classification:
            classification_score = 0.35
        elif (
            action.classification == Classification.NEEDS_ESCALATION
            and alert.ground_truth_classification == Classification.TRUE_POSITIVE
        ):
            # Escalating a true positive is partially correct
            classification_score = 0.20

        # 2. Priority accuracy (up to 0.25, scaled by distance)
        dist = abs(action.priority - alert.ground_truth_priority)
        priority_score = max(0.0, 0.25 * (1.0 - dist / 4.0))

        # 3. Justification quality (up to 0.15)
        if alert.relevant_iocs:
            justification_lower = action.justification.lower()
            matched = sum(
                1 for ioc in alert.relevant_iocs
                if ioc.lower() in justification_lower
            )
            justification_score = 0.15 * min(matched / max(len(alert.relevant_iocs), 1), 1.0)

        # 4. Disinformation resistance (hard task only, up to 0.20)
        if self._task_id == "hard":
            disinfo_reports = [r for r in self._intel_reports if r.is_disinformation]
            if disinfo_reports:
                justification_lower = action.justification.lower()
                # Bonus if agent explicitly notes disinformation / untrustworthiness
                disinfo_keywords = [
                    "disinformation", "misleading", "false report",
                    "adversarial", "planted", "untrusted", "do not trust",
                    "suspicious intel", "fabricated", "ignore this report",
                ]
                mentioned = any(kw in justification_lower for kw in disinfo_keywords)
                if mentioned:
                    disinformation_bonus = 0.20
                # Penalty if agent cites disinformation report as reason to classify TP as FP
                for dr in disinfo_reports:
                    if (
                        dr.id.lower() in justification_lower
                        and action.classification == Classification.FALSE_POSITIVE
                        and alert.ground_truth_classification == Classification.TRUE_POSITIVE
                    ):
                        disinformation_bonus = -0.15
                        break

        # 5. Time penalty — mild cost for using extra steps
        optimal_steps = len(self._alerts)
        if self._step_count > optimal_steps:
            time_penalty = -0.05 * (self._step_count - optimal_steps) / max(self._max_steps, 1)

        total = max(0.0, min(1.0,
            classification_score + priority_score + justification_score
            + disinformation_bonus + time_penalty
        ))

        return Reward(
            total=round(total, 4),
            classification_score=round(classification_score, 4),
            priority_score=round(priority_score, 4),
            justification_score=round(justification_score, 4),
            disinformation_bonus=round(disinformation_bonus, 4),
            time_penalty=round(time_penalty, 4),
            breakdown={
                "classification": round(classification_score, 4),
                "priority": round(priority_score, 4),
                "justification": round(justification_score, 4),
                "disinformation": round(disinformation_bonus, 4),
                "time": round(time_penalty, 4),
            },
        )

    # ── OpenEnv interface ────────────────────────────────────────────────

    def reset(self, task_id: str) -> Observation:
        """Load a task scenario and return the initial observation."""
        task = get_task(task_id)
        self._task_id = task_id
        self._task_meta = task
        self._alerts = copy.deepcopy(task["alerts"])
        self._intel_reports = copy.deepcopy(task["intel_reports"])
        self._pending_ids = [a.id for a in self._alerts]
        self._current_idx = 0
        self._step_count = 0
        self._max_steps = task["max_steps"]
        self._done = False
        self._scores = {}
        self._reward_details = {}
        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        """Process the agent's action and return the next observation."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step_count += 1

        # Validate alert_id
        alert = self._alert_by_id(action.alert_id)
        if alert is None:
            # Invalid alert — zero reward, don't advance
            return StepResult(
                observation=self._build_observation(),
                reward=0.0,
                done=self._done,
                info={"error": f"Unknown alert_id: {action.alert_id}"},
            )

        if action.alert_id not in self._pending_ids:
            return StepResult(
                observation=self._build_observation(),
                reward=0.0,
                done=self._done,
                info={"error": f"Alert {action.alert_id} already processed"},
            )

        # Compute reward
        reward = self._compute_reward(action, alert)
        self._scores[action.alert_id] = reward.total
        self._reward_details[action.alert_id] = reward.breakdown

        # Advance state
        self._pending_ids.remove(action.alert_id)
        self._current_idx += 1

        # Check done conditions
        if not self._pending_ids or self._step_count >= self._max_steps:
            self._done = True

        obs = self._build_observation()

        info: dict[str, Any] = {
            "reward_breakdown": reward.breakdown,
            "alert_id": action.alert_id,
        }
        if self._done:
            final_score = sum(self._scores.values()) / max(len(self._alerts), 1)
            info["final_score"] = round(final_score, 4)
            info["alerts_graded"] = len(self._scores)
            info["alerts_missed"] = len(self._pending_ids)

        return StepResult(
            observation=obs,
            reward=reward.total,
            done=self._done,
            info=info,
        )

    def state(self) -> EnvState:
        """Return the full internal state snapshot."""
        processed = len(self._scores)
        avg = sum(self._scores.values()) / max(processed, 1)
        return EnvState(
            task_id=self._task_id,
            step_count=self._step_count,
            max_steps=self._max_steps,
            alerts_processed=processed,
            alerts_total=len(self._alerts),
            current_score=round(avg, 4),
            done=self._done,
            scores_per_alert=self._scores,
            pending_alert_ids=list(self._pending_ids),
        )
