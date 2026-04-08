"""
CyberSentinel — Deterministic task graders.

Each grader compares agent classifications to ground truth and
returns a float score in [0.0, 1.0].
"""

from __future__ import annotations

from models import Action, Classification


def grade_task(
    actions: list[Action],
    alerts: list,  # list[Alert]
    task_id: str,
) -> dict:
    """
    Grade a complete task run.

    Returns a dict with:
      - score: float in [0.0, 1.0]
      - details: per-alert breakdown
      - summary: human-readable summary
    """
    alert_map = {a.id: a for a in alerts}
    total_alerts = len(alerts)

    if not actions:
        return {
            "score": 0.0,
            "details": {},
            "summary": "No actions taken — score 0.0",
        }

    per_alert: dict[str, dict] = {}
    total_score = 0.0
    correct = 0
    wrong = 0
    escalated = 0
    missed_ids = set(alert_map.keys())

    for action in actions:
        if action.alert_id not in alert_map:
            continue

        alert = alert_map[action.alert_id]
        missed_ids.discard(action.alert_id)

        # Classification accuracy (0.0 or 1.0)
        cls_correct = action.classification == alert.ground_truth_classification
        cls_score = 1.0 if cls_correct else 0.0

        # Partial credit for escalating a true positive
        if (
            not cls_correct
            and action.classification == Classification.NEEDS_ESCALATION
            and alert.ground_truth_classification == Classification.TRUE_POSITIVE
        ):
            cls_score = 0.6
            escalated += 1
        elif cls_correct:
            correct += 1
        else:
            wrong += 1

        # Priority accuracy (0.0 to 1.0)
        dist = abs(action.priority - alert.ground_truth_priority)
        pri_score = max(0.0, 1.0 - dist / 4.0)

        # Combined alert score (70% classification, 30% priority)
        alert_score = 0.7 * cls_score + 0.3 * pri_score

        per_alert[action.alert_id] = {
            "classification_correct": cls_correct,
            "classification_score": round(cls_score, 4),
            "priority_score": round(pri_score, 4),
            "combined_score": round(alert_score, 4),
        }
        total_score += alert_score

    # Penalise missed alerts (they count as 0)
    final_score = total_score / total_alerts if total_alerts > 0 else 0.0
    final_score = round(max(0.0, min(1.0, final_score)), 4)

    summary_parts = [
        f"Score: {final_score}",
        f"Correct: {correct}/{total_alerts}",
        f"Wrong: {wrong}",
        f"Escalated (partial credit): {escalated}",
        f"Missed: {len(missed_ids)}",
    ]

    return {
        "score": final_score,
        "details": per_alert,
        "missed": list(missed_ids),
        "summary": " | ".join(summary_parts),
    }
