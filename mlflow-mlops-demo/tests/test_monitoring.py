from datetime import datetime, timedelta, timezone

from northpeak_mlops.policy import DriftPolicy, decide_retraining


POLICY = DriftPolicy(
    warning_rmse_increase_pct=10.0,
    critical_rmse_increase_pct=20.0,
    consecutive_critical_periods=2,
    minimum_impressions=50_000,
    retrain_cooldown_days=7,
)
NOW = datetime(2026, 9, 8, tzinfo=timezone.utc)


def metric(status="critical", impressions=100_000):
    return {
        "metric_date": "2026-09-08",
        "weighted_rmse": 0.012,
        "baseline_rmse": 0.008,
        "rmse_change_pct": 50.0,
        "impressions": impressions,
        "drift_status": status,
    }


def test_persistent_critical_drift_triggers_retraining():
    decision = decide_retraining([metric(), metric()], None, POLICY, NOW)
    assert decision.should_retrain


def test_single_breach_does_not_trigger_retraining():
    decision = decide_retraining([metric()], None, POLICY, NOW)
    assert not decision.should_retrain


def test_low_volume_does_not_trigger_retraining():
    decision = decide_retraining([metric(impressions=10_000), metric()], None, POLICY, NOW)
    assert not decision.should_retrain


def test_cooldown_prevents_retraining():
    decision = decide_retraining([metric(), metric()], NOW - timedelta(days=2), POLICY, NOW)
    assert not decision.should_retrain


def test_warning_breaks_consecutive_critical_periods():
    decision = decide_retraining([metric(), metric(status="warning")], None, POLICY, NOW)
    assert not decision.should_retrain
