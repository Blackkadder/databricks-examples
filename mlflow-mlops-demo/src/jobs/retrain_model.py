"""Automated NorthPeak CTR retraining entrypoint.

This file intentionally contains no MLflow primitives. It demonstrates how a
job author consumes a small, reusable MLOps helper API.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Databricks executes workspace-file Python tasks through ``exec``, so ``__file__``
# is not defined. On current runtimes the working directory is the script folder;
# the additional candidates also keep direct local execution convenient.
working_directory = Path.cwd()
runtime_script = globals().get("__file__") or globals().get("filename")
source_candidates = []
if runtime_script:
    source_candidates.append(Path(runtime_script).resolve().parents[1])
source_candidates.extend([
    working_directory.parent,
    working_directory / "src",
    working_directory,
])
for source_root in source_candidates:
    if (source_root / "northpeak_mlops").is_dir():
        sys.path.insert(0, str(source_root))
        break

from pyspark.sql import SparkSession

from northpeak_mlops.features import load_training_frame
from northpeak_mlops.registry import deploy_model, register_model
from northpeak_mlops.scoring import batch_score_model, incumbent_drift_rmse
from northpeak_mlops.training import TrainingConfig, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain and conditionally deploy the NorthPeak CTR model")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--min-improvement-pct", type=float, default=0.0)
    parser.add_argument("--trials", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spark = SparkSession.builder.getOrCreate()
    model_name = f"{args.catalog}.{args.schema}.ctr_regressor"

    training_frame = load_training_frame(spark, args.catalog, args.schema)
    incumbent_rmse = incumbent_drift_rmse(spark, args.catalog, args.schema)
    training_result = train_model(
        training_frame,
        TrainingConfig(experiment_id=args.experiment_id, n_trials=args.trials),
    )
    version = register_model(training_result.model_uri, model_name)
    decision = deploy_model(
        model_name=model_name,
        version=version,
        candidate_rmse=training_result.drift_rmse,
        incumbent_rmse=incumbent_rmse,
        min_improvement_pct=args.min_improvement_pct,
    )

    rows_scored = 0
    if decision.promoted:
        rows_scored = batch_score_model(
            spark,
            training_frame,
            training_result,
            args.catalog,
            args.schema,
            version,
        )

    print(
        json.dumps(
            {
                "model_name": model_name,
                "candidate_version": version,
                "promoted": decision.promoted,
                "promotion_reason": decision.reason,
                "candidate_drift_rmse": round(training_result.drift_rmse, 6),
                "incumbent_drift_rmse": None if incumbent_rmse is None else round(incumbent_rmse, 6),
                "feature_set": training_result.feature_set_name,
                "rows_scored": rows_scored,
                "run_id": training_result.run_id,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
