"""Incumbent evaluation and governed batch prediction helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error

from .features import build_features
from .training import TrainingResult


def incumbent_drift_rmse(spark, catalog: str, schema: str) -> float | None:
    """Read the current champion's comparable drift metric from its prediction table."""
    table_name = f"{catalog}.{schema}.gold_ctr_predictions"
    try:
        frame = spark.table(table_name).where("is_drift_segment AND is_drift_window").toPandas()
    except Exception:
        return None
    if frame.empty:
        return None
    return float(
        np.sqrt(
            mean_squared_error(
                frame["actual_ctr"],
                frame["predicted_ctr_champion"],
                sample_weight=frame["impressions"],
            )
        )
    )


def batch_score_model(
    spark,
    frame,
    result: TrainingResult,
    catalog: str,
    schema: str,
    model_version: str,
) -> int:
    """Score the promoted candidate and replace the governed prediction table."""
    score = frame.copy()
    features = build_features(frame, result.feature_names, result.encoded_columns)
    score["predicted_ctr_champion"] = np.clip(result.model.predict(features), 0.0, 0.15)

    # Preserve the prior champion as the before/after baseline for the dashboard.
    prior = spark.table(f"{catalog}.{schema}.gold_ctr_predictions").select(
        "event_date",
        "channel",
        "device",
        "country",
        "visitor_type",
        "campaign_id",
        "predicted_ctr_champion",
    ).toPandas()
    prior = prior.rename(columns={"predicted_ctr_champion": "predicted_ctr_stale"})
    keys = ["event_date", "channel", "device", "country", "visitor_type", "campaign_id"]
    score = score.drop(columns=["predicted_ctr_stale"], errors="ignore").merge(prior, on=keys, how="left")
    score["ctr_gap_stale"] = score["predicted_ctr_stale"] - score["actual_ctr"]
    score["ctr_gap_champion"] = score["predicted_ctr_champion"] - score["actual_ctr"]
    score["model_version_champion"] = str(model_version)

    columns = [
        "event_date", "channel", "device", "country", "visitor_type", "campaign_id",
        "campaign_name", "objective", "impressions", "spend_usd", "actual_ctr",
        "predicted_ctr_stale", "predicted_ctr_champion", "ctr_gap_stale",
        "ctr_gap_champion", "is_drift_window", "is_drift_segment", "model_version_champion",
    ]
    output = score[columns].copy()
    output["is_drift_window"] = output["is_drift_window"].astype(bool)
    output["is_drift_segment"] = output["is_drift_segment"].astype(bool)
    spark_output = spark.createDataFrame(output)
    spark_output.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        f"{catalog}.{schema}.gold_ctr_predictions"
    )
    return len(output)
