"""Data loading and feature engineering shared by training and scoring."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .config import load_training_config


_CONFIG = load_training_config()
CATEGORICALS = tuple(_CONFIG["categorical_features"])
FEATURE_SETS = _CONFIG["feature_sets"]


def load_training_frame(spark, catalog: str, schema: str) -> pd.DataFrame:
    """Load the governed segment-day feature table into a small pandas frame."""
    frame = spark.table(f"{catalog}.{schema}.gold_segment_ctr_daily").toPandas()
    frame["event_date"] = pd.to_datetime(frame["event_date"])
    frame["day_of_week"] = frame["event_date"].dt.dayofweek.astype(int)
    frame["is_drift_window"] = frame["is_drift_window"].astype(int)
    for column in CATEGORICALS:
        frame[column] = frame[column].astype("category")
    return frame


def build_features(
    frame: pd.DataFrame,
    feature_names: Iterable[str],
    template_columns: list[str] | None = None,
) -> pd.DataFrame:
    """One-hot encode a named feature set and optionally align to a model schema."""
    feature_names = list(feature_names)
    categoricals = [name for name in feature_names if name in CATEGORICALS]
    numeric = [name for name in feature_names if name not in CATEGORICALS]
    encoded = pd.get_dummies(frame[categoricals + numeric].copy(), columns=categoricals, dtype=float)
    if template_columns is not None:
        encoded = encoded.reindex(columns=template_columns, fill_value=0.0)
    return encoded
