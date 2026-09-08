"""Load declarative training configuration shipped with the helper library."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@lru_cache(maxsize=1)
def load_training_config() -> dict[str, Any]:
    config_path = Path(__file__).with_name("training_config.yml")
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    required = {"categorical_features", "feature_sets", "xgboost_search_space"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Training configuration is missing: {', '.join(sorted(missing))}")
    return config
