"""
CyberSentinel — Deterministic task graders.

Each grader compares agent classifications to ground truth and
returns a float score in [0.0, 1.0].

All weights are centralised in GraderConfig — edit here if you want
to tune the scoring formula.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from models import Action, Classification


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GraderConfig:
    """Centralised scoring weights — mirrors environment.py reward weights."""

    # Per-alert component weights (must sum ≤ 1.0)
    classification_weight: float = 0.35   # correct TP/FP/escalation
    priority_weight: float = 0.25         # scaled by distance from ground truth
    justification_weight: float = 0.15    # IOC keyword coverage
    disinformation_weight: float = 0.20   # for alerts on disinformation tasks

    # Partial credit for escalating a true positive instead of classifying it
    escalation_partial_credit: float = 0.60

    # Penalty per missed alert (beyond the first free miss)
    missed_penalty: float = 0.0           # applied as negative contribution

    # Cross-alert correlation bonus (per alert whose justification cites a sibling)
    correlation_bonus: float = 0.10

    # Kill-chain bonus: awarded once if enough campaign alerts are correct
    chain_bonus: float = 0.10             # added to final score (clamped at 1.0)
    chain_bonus_threshold: int = 6        # min correct campaign alerts to trigger

    # Time penalty per excess step beyond optimal, capped at max_time_penalty
    time_penalty_per_step: float = 0.10
    max_time_penalty: float = 0.30

    # Disinformation detection keywords
    disinformation_keywords: list[str] = field(default_factory=lambda: [
        "disinformation", "misleading", "false report",
        "adversarial", "planted", "untrusted", "do not trust",
        "suspicious intel", "fabricated", "ignore this report",
        "inaccurate", "incorrect report", "cannot be trusted",
    ])


DEFAULT_CONFIG = GraderConfig()


# ── TypedDict return types ────────────────────────────────────────────────────

class AlertGradeDetail(TypedDict):
    classification_correct: bool
    classification_score: float
    priority_score: float
    justification_score: float
    disinformation_score: float
    correlation_score: float
    combined_score: float
    missed: bool


class GradeResult(TypedDict):
    score: float
    details: dict[str, AlertGradeDetail]
    missed: list[str]
    summary: str
    chain_bonus_awarded: bool


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ioc_keyword_score(justification: str, relevant_iocs: list[str], weight: float) -> float:
    """Score how many relevant IOC terms appear in the justification (case-insensitive)."""
    if not relevant_iocs:
        return 0.0
    jl = justification.lower()
    matched = sum(1 for ioc in relevant_iocs if ioc.lower() in jl)
    return weight * min(matched / max(len(relevant_iocs), 1), 1.0)


def _disinformation_score(
    justification: str,
    classification: Classification,
    ground_truth: Classification,
    is_hard_task: bool,
    disinfo_report_ids: list[str],
    config: GraderConfig,
) -> float:
    """
    Award disinformation-resistance credit if the agent correctly identified
    misleading intel AND didn't misclassify the alert because of it.
    """
    if not is_hard_task:
        return 0.0

    jl = justification.lower()
    mentioned = any(kw in jl for kw in config.disinformation_keywords)

    # Bonus: agent called out disinformation
    if mentioned and classification == ground_truth:
        return config.disinformation_weight

    # Penalty: agent cited a disinfo report ID and misclassified TP as FP
    for dr_id in disinfo_report_ids:
        if (
            dr_id.lower() in jl
            and classification == Classification.FALSE_POSITIVE
            and ground_truth == Classification.TRUE_POSITIVE
        ):
            return -0.15

    return 0.0


def _correlation_score(
    justification: str,
    chain_alert_ids: list[str],
    config: GraderConfig,
) -> float:
    """Bonus if the agent's justification explicitly references a sibling alert ID."""
    if not chain_alert_ids:
        return 0.0
    jl = justification.lower()
    cited = any(aid.lower() in jl for aid in chain_alert_ids)
    return config.correlation_bonus if cited else 0.0


# ── Main grader ───────────────────────────────────────────────────────────────

def grade_task(
    actions: list[Action],
    alerts: list,           # list[Alert]
    task_id: str,
    config: GraderConfig = DEFAULT_CONFIG,
    step_count: int = 0,
) -> GradeResult:
    """
    Grade a complete task run.

    Returns a GradeResult with:
      - score: float in [0.0, 1.0]
      - details: per-alert breakdown (including missed alerts)
      - missed: list of alert IDs not processed
      - summary: human-readable summary
      - chain_bonus_awarded: True if kill-chain bonus was triggered
    """
    alert_map = {a.id: a for a in alerts}
    total_alerts = len(alerts)
    is_hard_task = task_id == "hard"

    # Collect disinformation report IDs (only relevant for grading hard task)
    disinfo_report_ids: list[str] = []
    if hasattr(alerts[0] if alerts else None, "__class__"):
        # Access via task data — will be passed in if needed; default empty
        pass

    if not actions:
        details: dict[str, AlertGradeDetail] = {
            a.id: AlertGradeDetail(
                classification_correct=False,
                classification_score=0.0,
                priority_score=0.0,
                justification_score=0.0,
                disinformation_score=0.0,
                correlation_score=0.0,
                combined_score=0.0,
                missed=True,
            )
            for a in alerts
        }
        return GradeResult(
            score=0.0,
            details=details,
            missed=list(alert_map.keys()),
            summary="No actions taken — score 0.0",
            chain_bonus_awarded=False,
        )

    # Build action map (last action wins if duplicate)
    action_map: dict[str, Action] = {}
    for act in actions:
        if act.alert_id in alert_map:
            action_map[act.alert_id] = act

    missed_ids = set(alert_map.keys()) - set(action_map.keys())
    details = {}
    total_score = 0.0
    correct = 0
    wrong = 0
    escalated = 0
    chain_correct = 0

    # Identify kill-chain alert IDs for hard task
    hard_chain_ids = {
        "ALERT-H001", "ALERT-H002", "ALERT-H004", "ALERT-H006",
        "ALERT-H007", "ALERT-H009", "ALERT-H011", "ALERT-H013", "ALERT-H014",
    }

    for alert in alerts:
        aid = alert.id

        if aid in missed_ids:
            details[aid] = AlertGradeDetail(
                classification_correct=False,
                classification_score=0.0,
                priority_score=0.0,
                justification_score=0.0,
                disinformation_score=0.0,
                correlation_score=0.0,
                combined_score=0.0,
                missed=True,
            )
            continue

        action = action_map[aid]

        # 1. Classification (0.0, partial, or full)
        cls_correct = action.classification == alert.ground_truth_classification
        if cls_correct:
            cls_score = config.classification_weight
            correct += 1
            if is_hard_task and aid in hard_chain_ids:
                chain_correct += 1
        elif (
            action.classification == Classification.NEEDS_ESCALATION
            and alert.ground_truth_classification == Classification.TRUE_POSITIVE
        ):
            cls_score = config.classification_weight * config.escalation_partial_credit
            escalated += 1
        else:
            cls_score = 0.0
            wrong += 1

        # 2. Priority accuracy (0.0 to weight, scaled by distance)
        dist = abs(action.priority - alert.ground_truth_priority)
        pri_score = max(0.0, config.priority_weight * (1.0 - dist / 4.0))

        # 3. Justification quality — IOC keyword coverage
        just_score = _ioc_keyword_score(
            action.justification,
            alert.relevant_iocs,
            config.justification_weight,
        )

        # 4. Disinformation resistance (hard task only)
        disinfo_score = _disinformation_score(
            action.justification,
            action.classification,
            alert.ground_truth_classification,
            is_hard_task,
            disinfo_report_ids,
            config,
        )

        # 5. Cross-alert correlation bonus
        corr_score = _correlation_score(
            action.justification,
            alert.chain_alert_ids,
            config,
        )

        combined = max(0.0, min(1.0,
            cls_score + pri_score + just_score + disinfo_score + corr_score
        ))

        details[aid] = AlertGradeDetail(
            classification_correct=cls_correct,
            classification_score=round(cls_score, 4),
            priority_score=round(pri_score, 4),
            justification_score=round(just_score, 4),
            disinformation_score=round(disinfo_score, 4),
            correlation_score=round(corr_score, 4),
            combined_score=round(combined, 4),
            missed=False,
        )
        total_score += combined

    # Missed-alert penalty (beyond first free miss)
    if len(missed_ids) > 1 and config.missed_penalty < 0:
        penalty = config.missed_penalty * (len(missed_ids) - 1)
        total_score = max(0.0, total_score + penalty)

    # Normalise by total alert count (missed = 0 contribution)
    final_score = total_score / total_alerts if total_alerts > 0 else 0.0

    # Kill-chain bonus for hard task
    chain_bonus_awarded = False
    if is_hard_task and chain_correct >= config.chain_bonus_threshold:
        final_score = min(1.0, final_score + config.chain_bonus)
        chain_bonus_awarded = True

    final_score = round(max(0.0, min(1.0, final_score)), 4)

    summary_parts = [
        f"Score: {final_score}",
        f"Correct: {correct}/{total_alerts}",
        f"Wrong: {wrong}",
        f"Escalated (partial credit): {escalated}",
        f"Missed: {len(missed_ids)}",
    ]
    if chain_bonus_awarded:
        summary_parts.append(f"Kill-chain bonus: +{config.chain_bonus}")

    return GradeResult(
        score=final_score,
        details=details,
        missed=list(missed_ids),
        summary=" | ".join(summary_parts),
        chain_bonus_awarded=chain_bonus_awarded,
    )
