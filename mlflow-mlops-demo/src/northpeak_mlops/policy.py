"""Pure drift-policy types and decision logic."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class DriftPolicy:
    warning_rmse_increase_pct: float
    critical_rmse_increase_pct: float
    consecutive_critical_periods: int
    minimum_impressions: int
    retrain_cooldown_days: int


@dataclass(frozen=True)
class DriftDecision:
    should_retrain: bool
    reason: str
    metric_date: str | None
    rmse: float | None
    baseline_rmse: float | None
    rmse_change_pct: float | None
    impressions: int


def decide_retraining(
    recent_overall_metrics: list[dict],
    last_triggered_at: datetime | None,
    policy: DriftPolicy,
    now: datetime | None = None,
) -> DriftDecision:
    """Apply consecutive-breach, data-volume, and cooldown safeguards."""
    now = now or datetime.now(timezone.utc)
    latest = recent_overall_metrics[0] if recent_overall_metrics else {}

    def decision(should_retrain: bool, reason: str) -> DriftDecision:
        return DriftDecision(
            should_retrain=should_retrain,
            reason=reason,
            metric_date=str(latest.get("metric_date")) if latest else None,
            rmse=latest.get("weighted_rmse"),
            baseline_rmse=latest.get("baseline_rmse"),
            rmse_change_pct=latest.get("rmse_change_pct"),
            impressions=int(latest.get("impressions", 0)),
        )

    if len(recent_overall_metrics) < policy.consecutive_critical_periods:
        return decision(False, "insufficient metric history")
    if int(latest.get("impressions", 0)) < policy.minimum_impressions:
        return decision(False, "minimum impression volume not met")
    if last_triggered_at is not None:
        normalized = last_triggered_at.replace(tzinfo=last_triggered_at.tzinfo or timezone.utc)
        if now - normalized < timedelta(days=policy.retrain_cooldown_days):
            return decision(False, "retraining cooldown is active")
    required = recent_overall_metrics[: policy.consecutive_critical_periods]
    if not all(row.get("drift_status") == "critical" for row in required):
        return decision(False, "critical threshold has not persisted")
    return decision(True, "persistent critical drift exceeded the retraining threshold")
