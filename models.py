"""
CyberSentinel — Pydantic v2 typed models for the OpenEnv interface.

Defines the observation, action, reward, and internal-state schemas
that flow through reset() / step() / state().
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ── Enums ────────────────────────────────────────────────────────────────────

class AlertType(str, enum.Enum):
    IDS = "ids"
    FIREWALL = "firewall"
    ENDPOINT = "endpoint"
    EMAIL = "email"
    DNS = "dns"


class Severity(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Classification(str, enum.Enum):
    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"
    NEEDS_ESCALATION = "needs_escalation"


# ── Domain objects ───────────────────────────────────────────────────────────

class Alert(BaseModel):
    """A single security alert surfaced to the SOC analyst."""

    id: str = Field(..., description="Unique alert identifier")
    timestamp: str = Field(..., description="ISO-8601 timestamp")
    source_ip: str = Field(..., description="Source IP address")
    dest_ip: str = Field(..., description="Destination IP address")
    alert_type: AlertType
    severity_raw: Severity
    description: str = Field(..., description="Human-readable summary")
    raw_log: str = Field(..., description="Raw log line / packet excerpt")

    # Hidden ground-truth — never exposed in observations
    ground_truth_classification: Classification = Field(
        ..., exclude=True, description="Ground-truth label (hidden from agent)"
    )
    ground_truth_priority: int = Field(
        ..., ge=1, le=5, exclude=True,
        description="Ground-truth priority 1-5 (hidden from agent)"
    )
    relevant_iocs: list[str] = Field(
        default_factory=list, exclude=True,
        description="IOCs the justification should mention (hidden)"
    )
    # Optional: which sibling alert IDs form the same attack chain
    chain_alert_ids: list[str] = Field(
        default_factory=list, exclude=True,
        description="Alert IDs in the same attack chain (for correlation bonus)"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """Ensure timestamp looks like an ISO-8601 datetime."""
        import re
        pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
        if not re.match(pattern, v):
            raise ValueError(f"timestamp must be ISO-8601 format, got: {v!r}")
        return v


class ThreatIntelReport(BaseModel):
    """Threat intelligence report available for analyst context."""

    id: str
    source: str = Field(..., description="Intel provider name")
    title: str
    summary: str
    iocs: list[str] = Field(default_factory=list, description="Indicators of compromise")
    # Hidden ground-truth
    is_disinformation: bool = Field(
        False, exclude=True,
        description="True if this report was planted by the adversary"
    )


# ── OpenEnv interface models ────────────────────────────────────────────────

class Observation(BaseModel):
    """What the agent sees each step."""

    pending_alerts: list[dict[str, Any]] = Field(
        ..., description="Alerts still waiting for classification"
    )
    intel_reports: list[dict[str, Any]] = Field(
        ..., description="Available threat-intel reports"
    )
    current_alert: dict[str, Any] | None = Field(
        None, description="The alert the agent should classify this step"
    )
    time_remaining: int = Field(..., description="Steps remaining")
    alerts_processed: int = Field(0, description="Alerts classified so far")
    alerts_total: int = Field(..., description="Total alerts in the task")
    running_reward_avg: float = Field(
        0.0,
        description="Running average of per-step rewards (in-episode signal, not final score)"
    )
    steps_warning: bool = Field(
        False,
        description="True when 3 or fewer steps remain — agent should prioritise"
    )


class Action(BaseModel):
    """Agent's response to the current alert."""

    alert_id: str = Field(..., description="ID of the alert being classified")
    classification: Classification
    priority: int = Field(..., ge=1, le=5, description="Agent-assigned priority 1-5")
    justification: str = Field(
        ..., min_length=1, max_length=2000,
        description="Free-text justification for the classification"
    )


class StepResult(BaseModel):
    """Return value of step()."""

    observation: Observation
    reward: float = Field(..., ge=0.0, le=1.0)
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class Reward(BaseModel):
    """Detailed reward breakdown."""

    total: float = Field(..., ge=0.0, le=1.0)
    classification_score: float = 0.0
    priority_score: float = 0.0
    justification_score: float = 0.0
    disinformation_bonus: float = 0.0
    correlation_bonus: float = 0.0
    time_penalty: float = 0.0
    breakdown: dict[str, float] = Field(default_factory=dict)


class EnvState(BaseModel):
    """Full internal state returned by state()."""

    task_id: str
    step_count: int
    max_steps: int
    alerts_processed: int
    alerts_total: int
    current_score: float
    done: bool
    scores_per_alert: dict[str, float] = Field(default_factory=dict)
    reward_details_per_alert: dict[str, dict] = Field(default_factory=dict)
    pending_alert_ids: list[str] = Field(default_factory=list)
