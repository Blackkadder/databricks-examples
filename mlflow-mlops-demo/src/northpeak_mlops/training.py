"""MLflow-backed model training hidden behind a small application API."""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
import numpy as np
import optuna
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from .config import load_training_config
from .features import FEATURE_SETS, build_features


@dataclass(frozen=True)
class TrainingConfig:
    experiment_id: str
    n_trials: int = 4
    random_seed: int = 42


@dataclass
class TrainingResult:
    model_uri: str
    run_id: str
    model: XGBRegressor
    feature_set_name: str
    feature_names: list[str]
    encoded_columns: list[str]
    validation_rmse: float
    drift_rmse: float
    top_features: list[str]


def _configure_tracking(experiment_id: str) -> None:
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(experiment_id=experiment_id)
    mlflow.xgboost.autolog(log_input_examples=False, log_models=False, silent=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_model(frame: pd.DataFrame, config: TrainingConfig) -> TrainingResult:
    """Tune feature sets, log candidates, and return one logged final model."""
    _configure_tracking(config.experiment_id)
    target = frame["actual_ctr"].to_numpy()
    weights = frame["impressions"].to_numpy()
    train_weights = weights * np.where(frame["is_drift_window"].to_numpy() == 1, 4.0, 1.0)
    train_idx, validation_idx = train_test_split(
        frame.index, test_size=0.2, random_state=config.random_seed
    )
    cohort = (
        frame.loc[validation_idx, "is_drift_segment"].astype(bool)
        & frame.loc[validation_idx, "is_drift_window"].astype(bool)
    ).to_numpy()

    search_space = load_training_config()["xgboost_search_space"]
    candidates: list[dict] = []
    for offset, (feature_set_name, feature_names) in enumerate(FEATURE_SETS.items()):
        features = build_features(frame, feature_names)
        x_train, x_validation = features.loc[train_idx], features.loc[validation_idx]
        y_train, y_validation = target[train_idx], target[validation_idx]
        w_train, w_validation = train_weights[train_idx], weights[validation_idx]

        def objective(trial):
            params = {}
            for name, definition in search_space.items():
                parameter_type = definition["type"]
                if parameter_type == "int":
                    params[name] = trial.suggest_int(name, definition["low"], definition["high"])
                elif parameter_type == "float":
                    params[name] = trial.suggest_float(
                        name,
                        definition["low"],
                        definition["high"],
                        log=definition.get("log", False),
                    )
                else:
                    raise ValueError(f"Unsupported search-space type {parameter_type!r} for {name}")
            with mlflow.start_run(nested=True):
                mlflow.set_tags({"model_generation": "retrain_candidate", "feature_set": feature_set_name})
                model = XGBRegressor(**params, random_state=config.random_seed)
                model.fit(x_train, y_train, sample_weight=w_train)
                predictions = model.predict(x_validation)
                rmse = float(
                    np.sqrt(
                        mean_squared_error(
                            y_validation[cohort], predictions[cohort], sample_weight=w_validation[cohort]
                        )
                    )
                )
                mlflow.log_metrics(
                    {
                        "rmse_misbid_cohort": rmse,
                        "rmse": float(np.sqrt(mean_squared_error(y_validation, predictions, sample_weight=w_validation))),
                        "mae": float(mean_absolute_error(y_validation, predictions, sample_weight=w_validation)),
                        "r2": float(r2_score(y_validation, predictions, sample_weight=w_validation)),
                    }
                )
                return rmse

        sampler = optuna.samplers.TPESampler(seed=config.random_seed + offset)
        with mlflow.start_run(run_name=f"automated_hpo_{feature_set_name}"):
            study = optuna.create_study(direction="minimize", sampler=sampler)
            study.optimize(objective, n_trials=config.n_trials)
        candidates.append(
            {
                "feature_set_name": feature_set_name,
                "feature_names": feature_names,
                "best_params": study.best_params,
                "cohort_rmse": study.best_value,
            }
        )

    winner = min(candidates, key=lambda candidate: candidate["cohort_rmse"])
    features = build_features(frame, winner["feature_names"])
    (
        x_train,
        x_validation,
        y_train,
        y_validation,
        w_train,
        _w_validation_train,
        _w_train_base,
        w_validation,
    ) = train_test_split(
        features,
        target,
        train_weights,
        weights,
        test_size=0.2,
        random_state=config.random_seed,
    )

    with mlflow.start_run(run_name="automated_retrain_candidate") as run:
        mlflow.set_tags(
            {"model_generation": "automated_retrain", "feature_set": winner["feature_set_name"]}
        )
        mlflow.log_params(winner["best_params"])
        model = XGBRegressor(**winner["best_params"], random_state=config.random_seed)
        model.fit(x_train, y_train, sample_weight=w_train)
        validation_predictions = model.predict(x_validation)
        validation_rmse = float(
            np.sqrt(mean_squared_error(y_validation, validation_predictions, sample_weight=w_validation))
        )
        mlflow.log_metric("rmse", validation_rmse)

        drift_mask = frame["is_drift_segment"].astype(bool) & frame["is_drift_window"].astype(bool)
        drift_predictions = model.predict(features.loc[drift_mask])
        drift_rmse = float(
            np.sqrt(
                mean_squared_error(
                    frame.loc[drift_mask, "actual_ctr"],
                    drift_predictions,
                    sample_weight=frame.loc[drift_mask, "impressions"],
                )
            )
        )
        mlflow.log_metric("rmse_misbid_cohort", drift_rmse)

        importance = pd.DataFrame(
            {"feature": list(features.columns), "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)
        artifact_path = "/tmp/automated_retrain_feature_importance.csv"
        importance.to_csv(artifact_path, index=False)
        mlflow.log_artifact(artifact_path)
        model_info = mlflow.xgboost.log_model(
            model,
            name="model",
            signature=infer_signature(x_validation, validation_predictions),
            input_example=x_validation.head(3),
        )

    return TrainingResult(
        model_uri=model_info.model_uri,
        run_id=run.info.run_id,
        model=model,
        feature_set_name=winner["feature_set_name"],
        feature_names=list(winner["feature_names"]),
        encoded_columns=list(features.columns),
        validation_rmse=validation_rmse,
        drift_rmse=drift_rmse,
        top_features=importance.head(10)["feature"].tolist(),
    )
