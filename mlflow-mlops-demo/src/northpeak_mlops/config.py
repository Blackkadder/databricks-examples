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

    required = {"destination", "categorical_features", "feature_sets", "xgboost_search_space"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Training configuration is missing: {', '.join(sorted(missing))}")

    required_destination = {"catalog", "schema", "tables", "registered_model", "experiments"}
    missing_destination = required_destination.difference(config["destination"])
    if missing_destination:
        raise ValueError(
            "Destination configuration is missing: "
            f"{', '.join(sorted(missing_destination))}"
        )
    return config
