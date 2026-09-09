"""Compute granular model-health metrics and decide whether to retrain."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timezone
from pathlib import Path

working_directory = Path.cwd()
runtime_script = globals().get("__file__") or globals().get("filename")
source_candidates = []
if runtime_script:
    source_candidates.append(Path(runtime_script).resolve().parents[1])
source_candidates.extend([working_directory.parent, working_directory / "src", working_directory])
for source_root in source_candidates:
    if (source_root / "northpeak_mlops").is_dir():
        sys.path.insert(0, str(source_root))
        break

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from northpeak_mlops.config import load_training_config
from northpeak_mlops.monitoring import build_monitoring_metrics
from northpeak_mlops.policy import DriftPolicy, decide_retraining


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor NorthPeak CTR model drift")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spark = SparkSession.builder.getOrCreate()
    config = load_training_config()["monitoring"]
    predictions_table = f"{args.catalog}.{args.schema}.gold_ctr_predictions"
    metrics_table = f"{args.catalog}.{args.schema}.gold_model_monitoring_metrics"
    decisions_table = f"{args.catalog}.{args.schema}.model_retrain_decisions"

    metrics = build_monitoring_metrics(
        spark.table(predictions_table),
        windows_days=config["windows_days"],
        dimensions=config["dimensions"],
        warning_pct=config["warning_rmse_increase_pct"],
        critical_pct=config["critical_rmse_increase_pct"],
    )
    metrics.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(metrics_table)

    recent = [row.asDict(recursive=True) for row in (
        metrics.where(
            (F.col("dimension_name") == "overall")
            & (F.col("dimension_value") == "All")
            & (F.col("window_days") == 7)
        ).orderBy(F.col("metric_date").desc()).limit(config["consecutive_critical_periods"]).collect()
    )]
    last_triggered_at = None
    if spark.catalog.tableExists(decisions_table):
        last_row = (
            spark.table(decisions_table).where("should_retrain")
            .orderBy(F.col("evaluated_at").desc()).select("evaluated_at").first()
        )
        if last_row:
            last_triggered_at = last_row["evaluated_at"].replace(tzinfo=timezone.utc)

    policy = DriftPolicy(
        warning_rmse_increase_pct=config["warning_rmse_increase_pct"],
        critical_rmse_increase_pct=config["critical_rmse_increase_pct"],
        consecutive_critical_periods=config["consecutive_critical_periods"],
        minimum_impressions=config["minimum_impressions"],
        retrain_cooldown_days=config["retrain_cooldown_days"],
    )
    decision = decide_retraining(recent, last_triggered_at, policy)
    decision_row = spark.createDataFrame(
        [(
            decision.metric_date, decision.rmse, decision.baseline_rmse,
            decision.rmse_change_pct, decision.impressions,
            decision.should_retrain, decision.reason,
        )],
        "metric_date string, weighted_rmse double, baseline_rmse double, "
        "rmse_change_pct double, impressions long, should_retrain boolean, reason string",
    ).withColumn("evaluated_at", F.current_timestamp())
    decision_row.write.mode("append").option("mergeSchema", "true").saveAsTable(decisions_table)

    dbutils.jobs.taskValues.set(key="should_retrain", value=str(decision.should_retrain).lower())
    dbutils.jobs.taskValues.set(key="decision_reason", value=decision.reason)
    print(json.dumps(decision.__dict__, indent=2))


if __name__ == "__main__":
    main()
