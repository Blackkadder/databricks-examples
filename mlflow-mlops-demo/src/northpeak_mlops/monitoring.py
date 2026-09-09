"""Model-quality metrics and drift decisions shared by jobs and dashboards."""

from __future__ import annotations

from functools import reduce

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

def build_monitoring_metrics(
    predictions: DataFrame,
    windows_days: list[int],
    dimensions: list[str],
    warning_pct: float,
    critical_pct: float,
) -> DataFrame:
    """Build daily and rolling, impression-weighted accuracy metrics by slice."""
    dimension_pairs = [F.struct(F.lit("overall").alias("name"), F.lit("All").alias("value"))]
    dimension_pairs.extend(
        F.struct(F.lit(name).alias("name"), F.coalesce(F.col(name).cast("string"), F.lit("Unknown")).alias("value"))
        for name in dimensions
    )
    expanded = (
        predictions
        .withColumn("metric_date", F.to_date("event_date"))
        .withColumn("slice", F.explode(F.array(*dimension_pairs)))
        .select(
            "metric_date",
            F.col("slice.name").alias("dimension_name"),
            F.col("slice.value").alias("dimension_value"),
            F.col("model_version_champion").alias("model_version"),
            "impressions",
            "actual_ctr",
            "predicted_ctr_champion",
            "is_drift_window",
        )
    )
    error = F.col("predicted_ctr_champion") - F.col("actual_ctr")
    components = (
        expanded
        .withColumn("weighted_squared_error", F.pow(error, 2) * F.col("impressions"))
        .withColumn("weighted_absolute_error", F.abs(error) * F.col("impressions"))
        .withColumn("weighted_signed_error", error * F.col("impressions"))
        .withColumn("weighted_actual", F.col("actual_ctr") * F.col("impressions"))
        .withColumn("weighted_predicted", F.col("predicted_ctr_champion") * F.col("impressions"))
    )
    group_columns = ["metric_date", "dimension_name", "dimension_value", "model_version"]
    daily = components.groupBy(*group_columns).agg(
        F.sum("weighted_squared_error").alias("sum_squared_error"),
        F.sum("weighted_absolute_error").alias("sum_absolute_error"),
        F.sum("weighted_signed_error").alias("sum_signed_error"),
        F.sum("weighted_actual").alias("sum_actual"),
        F.sum("weighted_predicted").alias("sum_predicted"),
        F.sum("impressions").alias("impressions"),
        F.count(F.lit(1)).alias("observation_count"),
    )

    baseline = (
        components.where(~F.col("is_drift_window"))
        .groupBy("dimension_name", "dimension_value", "model_version")
        .agg(
            F.sqrt(F.sum("weighted_squared_error") / F.sum("impressions")).alias("baseline_rmse")
        )
    )

    rolling_frames = []
    for days in windows_days:
        seconds = max(days - 1, 0) * 86400
        window = (
            Window.partitionBy("dimension_name", "dimension_value", "model_version")
            .orderBy(F.col("metric_date").cast("timestamp").cast("long"))
            .rangeBetween(-seconds, 0)
        )
        rolling_frames.append(
            daily.select(
                "metric_date", "dimension_name", "dimension_value", "model_version",
                *[F.sum(name).over(window).alias(name) for name in (
                    "sum_squared_error", "sum_absolute_error", "sum_signed_error",
                    "sum_actual", "sum_predicted", "impressions", "observation_count",
                )],
            ).withColumn("window_days", F.lit(days))
        )

    metrics = reduce(DataFrame.unionByName, rolling_frames).join(
        baseline, ["dimension_name", "dimension_value", "model_version"], "left"
    )
    metrics = (
        metrics
        .withColumn("weighted_rmse", F.sqrt(F.col("sum_squared_error") / F.col("impressions")))
        .withColumn("weighted_mae", F.col("sum_absolute_error") / F.col("impressions"))
        .withColumn("signed_bias", F.col("sum_signed_error") / F.col("impressions"))
        .withColumn("actual_ctr", F.col("sum_actual") / F.col("impressions"))
        .withColumn("predicted_ctr", F.col("sum_predicted") / F.col("impressions"))
        .withColumn("absolute_ctr_gap", F.abs(F.col("predicted_ctr") - F.col("actual_ctr")))
        .withColumn(
            "rmse_change_pct",
            F.when(
                F.col("baseline_rmse") > 0,
                100.0 * (F.col("weighted_rmse") - F.col("baseline_rmse")) / F.col("baseline_rmse"),
            ),
        )
        .withColumn(
            "drift_status",
            F.when(F.col("rmse_change_pct") >= critical_pct, F.lit("critical"))
            .when(F.col("rmse_change_pct") >= warning_pct, F.lit("warning"))
            .otherwise(F.lit("healthy")),
        )
        .withColumn("computed_at", F.current_timestamp())
    )
    return metrics.select(
        "metric_date", "window_days", "dimension_name", "dimension_value", "model_version",
        "actual_ctr", "predicted_ctr", "weighted_rmse", "weighted_mae", "signed_bias",
        "absolute_ctr_gap", "baseline_rmse", "rmse_change_pct", "drift_status",
        "observation_count", "impressions", "computed_at",
    )
