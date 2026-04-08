"""
CyberSentinel — Smoke tests for the environment core.

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

    assert obs.alerts_total == 5, f"Expected 5 alerts, got {obs.alerts_total}"
    assert obs.current_alert is not None, "Expected a current_alert"
    assert obs.time_remaining == 10

    task = get_task("easy")
    gt = {a.id: a for a in task["alerts"]}

    step = 0
    while obs.current_alert is not None:
        alert_id = obs.current_alert["id"]
        ground = gt[alert_id]

        # Build a "perfect" action
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
    assert obs.alerts_total == 10
    assert obs.current_alert is not None
    print(f"  ✓ medium: loaded, {obs.alerts_total} alerts")


def test_hard_task_loads():
    """Verify hard task loads with disinformation reports."""
    from environment import CyberSentinelEnv
    from tasks import get_task

    env = CyberSentinelEnv()
    obs = env.reset("hard")
    assert obs.alerts_total == 15
    assert len(obs.intel_reports) == 6

    task = get_task("hard")
    disinfo = [r for r in task["intel_reports"] if r.is_disinformation]
    assert len(disinfo) == 2, f"Expected 2 disinformation reports, got {len(disinfo)}"

    # Verify disinformation flag is NOT exposed in observations
    for r in obs.intel_reports:
        assert "is_disinformation" not in r, "is_disinformation leaked to observation!"

    print(f"  ✓ hard: loaded, {obs.alerts_total} alerts, {len(obs.intel_reports)} intel reports (2 adversarial)")


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

    # Try same alert again
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

    # Process all alerts
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

    # Try to step again
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
    """Verify state() returns a valid EnvState."""
    from environment import CyberSentinelEnv

    env = CyberSentinelEnv()
    env.reset("easy")
    state = env.state()
    assert state.task_id == "easy"
    assert state.step_count == 0
    assert state.max_steps == 10
    assert state.done is False
    print("  ✓ state: returns valid EnvState")


def test_grader():
    """Verify grader produces correct scores."""
    from graders import grade_task
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
            justification="test",
        )
        for a in alerts
    ]
    result = grade_task(perfect_actions, alerts, "easy")
    assert result["score"] == 1.0, f"Perfect play should score 1.0, got {result['score']}"
    print(f"  ✓ grader-perfect: score={result['score']}")

    # All wrong actions
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
    print(f"  ✓ grader-wrong: score={result2['score']}")


# ── Runner ───────────────────────────────────────────────────────────────────

def main():
    tests = [
        test_easy_task_full_run,
        test_medium_task_loads,
        test_hard_task_loads,
        test_invalid_action,
        test_double_classification,
        test_done_then_step,
        test_state_endpoint,
        test_grader,
    ]

    print("CyberSentinel — Smoke Tests")
    print("=" * 50)

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

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    else:
        print("All tests passed! ✓")


if __name__ == "__main__":
    main()
