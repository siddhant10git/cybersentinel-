"""
CyberSentinel — Smoke tests for the environment core (v2).

Run with:  python test_env.py
"""

from __future__ import annotations

import sys
import traceback


def test_easy_task_full_run():
    """Run through the easy task classifying every alert correctly."""
    from environment import CyberSentinelEnv
    from models import Action, Classification
    from tasks import get_task

    env = CyberSentinelEnv()
    obs = env.reset("easy")

    assert obs.alerts_total == 7, f"Expected 7 alerts, got {obs.alerts_total}"
    assert obs.current_alert is not None, "Expected a current_alert"
    assert obs.time_remaining == 12

    task = get_task("easy")
    gt = {a.id: a for a in task["alerts"]}

    step = 0
    while obs.current_alert is not None:
        alert_id = obs.current_alert["id"]
        ground = gt[alert_id]

        action = Action(
            alert_id=alert_id,
            classification=ground.ground_truth_classification,
            priority=ground.ground_truth_priority,
            justification="IOCs: " + ", ".join(ground.relevant_iocs) if ground.relevant_iocs else "Benign activity.",
        )

        result = env.step(action)
        step += 1

        assert 0.0 <= result.reward <= 1.0, f"Reward {result.reward} out of range"

        if result.done:
            assert "final_score" in result.info
            score = result.info["final_score"]
            assert score > 0.5, f"Perfect play should score > 0.5, got {score}"
            print(f"  ✓ easy: {step} steps, final_score={score:.4f}")
            break

        obs = result.observation

    state = env.state()
    assert state.done is True
    assert state.task_id == "easy"


def test_medium_task_loads():
    """Verify medium task loads and has correct shape."""
    from environment import CyberSentinelEnv

    env = CyberSentinelEnv()
    obs = env.reset("medium")
    assert obs.alerts_total == 12, f"Expected 12 alerts, got {obs.alerts_total}"
    assert obs.current_alert is not None
    print(f"  ✓ medium: loaded, {obs.alerts_total} alerts")


def test_hard_task_loads():
    """Verify hard task loads with 3 disinformation reports."""
    from environment import CyberSentinelEnv
    from tasks import get_task

    env = CyberSentinelEnv()
    obs = env.reset("hard")
    assert obs.alerts_total == 17, f"Expected 17 alerts, got {obs.alerts_total}"
    assert len(obs.intel_reports) == 7, f"Expected 7 intel reports, got {len(obs.intel_reports)}"

    task = get_task("hard")
    disinfo = [r for r in task["intel_reports"] if r.is_disinformation]
    assert len(disinfo) == 3, f"Expected 3 disinformation reports, got {len(disinfo)}"

    # Verify disinformation flag is NOT exposed in observations
    for r in obs.intel_reports:
        assert "is_disinformation" not in r, "is_disinformation leaked to observation!"

    print(f"  ✓ hard: loaded, {obs.alerts_total} alerts, {len(obs.intel_reports)} intel reports (3 adversarial)")


def test_insider_task_loads():
    """Verify insider threat task loads correctly."""
    from environment import CyberSentinelEnv

    env = CyberSentinelEnv()
    obs = env.reset("insider")
    assert obs.alerts_total == 8, f"Expected 8 alerts, got {obs.alerts_total}"
    assert obs.time_remaining == 14
    print(f"  ✓ insider: loaded, {obs.alerts_total} alerts")


def test_zero_task_all_fp():
    """Verify zero alert shift is all false positives."""
    from tasks import get_task
    from models import Classification

    task = get_task("zero")
    for alert in task["alerts"]:
        assert alert.ground_truth_classification == Classification.FALSE_POSITIVE, \
            f"Expected all FP in zero task, got {alert.id}={alert.ground_truth_classification}"
    print(f"  ✓ zero: all {len(task['alerts'])} alerts confirmed FALSE_POSITIVE")


def test_invalid_action():
    """Verify invalid alert_id returns zero reward."""
    from environment import CyberSentinelEnv
    from models import Action, Classification

    env = CyberSentinelEnv()
    env.reset("easy")

    action = Action(
        alert_id="NONEXISTENT",
        classification=Classification.FALSE_POSITIVE,
        priority=1,
        justification="test",
    )
    result = env.step(action)
    assert result.reward == 0.0
    assert "error" in result.info
    print("  ✓ invalid action: handled correctly")


def test_double_classification():
    """Verify classifying the same alert twice returns zero reward."""
    from environment import CyberSentinelEnv
    from models import Action, Classification

    env = CyberSentinelEnv()
    obs = env.reset("easy")
    alert_id = obs.current_alert["id"]

    action = Action(
        alert_id=alert_id,
        classification=Classification.FALSE_POSITIVE,
        priority=1,
        justification="first classification",
    )
    env.step(action)

    action2 = Action(
        alert_id=alert_id,
        classification=Classification.TRUE_POSITIVE,
        priority=5,
        justification="duplicate",
    )
    result2 = env.step(action2)
    assert result2.reward == 0.0
    assert "error" in result2.info
    print("  ✓ double classification: rejected correctly")


def test_done_then_step():
    """Verify stepping after done raises an error."""
    from environment import CyberSentinelEnv
    from models import Action, Classification
    from tasks import get_task

    env = CyberSentinelEnv()
    obs = env.reset("easy")
    task = get_task("easy")
    gt = {a.id: a for a in task["alerts"]}

    while obs.current_alert is not None:
        aid = obs.current_alert["id"]
        ground = gt[aid]
        action = Action(
            alert_id=aid,
            classification=ground.ground_truth_classification,
            priority=ground.ground_truth_priority,
            justification="test",
        )
        result = env.step(action)
        if result.done:
            break
        obs = result.observation

    try:
        env.step(Action(
            alert_id="ALERT-E001",
            classification=Classification.FALSE_POSITIVE,
            priority=1,
            justification="should fail",
        ))
        assert False, "Expected RuntimeError"
    except RuntimeError:
        print("  ✓ step-after-done: raises RuntimeError")


def test_state_endpoint():
    """Verify state() returns a valid EnvState with reward_details."""
    from environment import CyberSentinelEnv

    env = CyberSentinelEnv()
    env.reset("easy")
    state = env.state()
    assert state.task_id == "easy"
    assert state.step_count == 0
    assert state.max_steps == 12
    assert state.done is False
    # After reset, reward_details_per_alert should be empty
    assert state.reward_details_per_alert == {}
    print("  ✓ state: returns valid EnvState with reward_details_per_alert")


def test_steps_warning():
    """Verify steps_warning fires when few steps remain."""
    from environment import CyberSentinelEnv
    from models import Action, Classification
    from tasks import get_task

    env = CyberSentinelEnv()
    obs = env.reset("zero")  # 5 alerts, max_steps=8
    task = get_task("zero")
    gt = {a.id: a for a in task["alerts"]}

    # Process all 5 alerts — check both the obs BEFORE and AFTER each step
    step_warnings_seen = []
    while obs.current_alert is not None:
        step_warnings_seen.append(obs.steps_warning)
        aid = obs.current_alert["id"]
        ground = gt[aid]
        action = Action(
            alert_id=aid,
            classification=ground.ground_truth_classification,
            priority=ground.ground_truth_priority,
            justification="Routine benign activity.",
        )
        result = env.step(action)
        # Also capture warning from result observation
        step_warnings_seen.append(result.observation.steps_warning)
        if result.done:
            break
        obs = result.observation

    # With max_steps=8 and 5 alerts, remaining=3 at step 5, so warning fires
    # The observation returned AFTER step 5 has time_remaining=3 → steps_warning=True
    assert any(step_warnings_seen), (
        f"steps_warning should fire at some point. warnings={step_warnings_seen}"
    )
    print(f"  [OK] steps_warning: fired correctly, warnings={step_warnings_seen}")


def test_grader():
    """Verify grader produces correct scores and includes missed alert details."""
    from graders import grade_task, GraderConfig
    from models import Action, Classification
    from tasks import get_task

    task = get_task("easy")
    alerts = task["alerts"]

    # Perfect actions
    perfect_actions = [
        Action(
            alert_id=a.id,
            classification=a.ground_truth_classification,
            priority=a.ground_truth_priority,
            justification="IOCs: " + ", ".join(a.relevant_iocs) if a.relevant_iocs else "Benign.",
        )
        for a in alerts
    ]
    result = grade_task(perfect_actions, alerts, "easy")
    assert result["score"] > 0.6, f"Perfect play (with IOC bonus) should score > 0.6, got {result['score']}"
    assert not result["chain_bonus_awarded"], "No chain bonus for easy task"
    print(f"  ✓ grader-perfect: score={result['score']}")

    # All wrong actions — missed alerts included in details
    wrong_actions = [
        Action(
            alert_id=a.id,
            classification=Classification.FALSE_POSITIVE
            if a.ground_truth_classification == Classification.TRUE_POSITIVE
            else Classification.TRUE_POSITIVE,
            priority=1 if a.ground_truth_priority == 5 else 5,
            justification="wrong",
        )
        for a in alerts
    ]
    result2 = grade_task(wrong_actions, alerts, "easy")
    assert result2["score"] < 0.5, f"All-wrong should score < 0.5, got {result2['score']}"
    # All alerts should be in details (none missed in wrong_actions)
    assert len(result2["details"]) == len(alerts)
    print(f"  ✓ grader-wrong: score={result2['score']}")

    # No actions — all missed
    result3 = grade_task([], alerts, "easy")
    assert result3["score"] == 0.0
    assert result3["missed"] == [a.id for a in alerts]
    assert all(d["missed"] for d in result3["details"].values())
    print(f"  ✓ grader-empty: score=0.0, all {len(alerts)} alerts in details as missed=True")


def test_chain_bonus_hard():
    """Verify kill-chain bonus fires on hard task with ≥6 chain alerts correct."""
    from graders import grade_task, GraderConfig
    from models import Action, Classification
    from tasks import get_task

    task = get_task("hard")
    alerts = task["alerts"]
    gt = {a.id: a for a in alerts}

    # Perfect actions
    perfect_actions = [
        Action(
            alert_id=a.id,
            classification=a.ground_truth_classification,
            priority=a.ground_truth_priority,
            justification="IOCs: " + ", ".join(a.relevant_iocs) if a.relevant_iocs else "Benign.",
        )
        for a in alerts
    ]
    result = grade_task(perfect_actions, alerts, "hard")
    assert result["chain_bonus_awarded"], "Kill-chain bonus should fire with perfect play"
    print(f"  ✓ chain-bonus (hard): awarded=True, score={result['score']}")


def test_cross_alert_correlation():
    """Verify correlation bonus fires when justification mentions a sibling alert."""
    from graders import grade_task
    from models import Action, Classification
    from tasks import get_task

    task = get_task("medium")
    alerts = task["alerts"]
    gt = {a.id: a for a in alerts}

    # Action for ALERT-M007 that cites sibling ALERT-M003
    m007 = gt["ALERT-M007"]
    action_with_correlation = Action(
        alert_id="ALERT-M007",
        classification=m007.ground_truth_classification,
        priority=m007.ground_truth_priority,
        justification="Large exfil from 10.0.1.10. Related to ALERT-M003 DGA C2 contact and ALERT-M001 SQLi.",
    )

    # Only grade this single alert
    result = grade_task([action_with_correlation], [m007], "medium")
    detail = result["details"].get("ALERT-M007", {})
    assert detail.get("correlation_score", 0) > 0, "Correlation bonus should fire"
    print(f"  ✓ cross-alert correlation: bonus={detail['correlation_score']}")


# ── Runner ───────────────────────────────────────────────────────────────────

def main():
    tests = [
        test_easy_task_full_run,
        test_medium_task_loads,
        test_hard_task_loads,
        test_insider_task_loads,
        test_zero_task_all_fp,
        test_invalid_action,
        test_double_classification,
        test_done_then_step,
        test_state_endpoint,
        test_steps_warning,
        test_grader,
        test_chain_bonus_hard,
        test_cross_alert_correlation,
    ]

    print("CyberSentinel v2 — Smoke Tests")
    print("=" * 55)

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception:
            failed += 1
            print(f"  ✗ {test.__name__}: FAILED")
            traceback.print_exc()
            print()

    print("=" * 55)
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    else:
        print("All tests passed! ✓")


if __name__ == "__main__":
    main()
