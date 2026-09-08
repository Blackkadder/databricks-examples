"""Unity Catalog registration and guarded deployment helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeploymentDecision:
    promoted: bool
    version: str
    candidate_rmse: float
    incumbent_rmse: float | None
    reason: str


def should_promote(
    candidate_rmse: float,
    incumbent_rmse: float | None,
    min_improvement_pct: float,
) -> tuple[bool, str]:
    """Return a deterministic quality-gate decision and explanation."""
    if incumbent_rmse is None:
        return True, "No incumbent prediction metric exists"
    required_rmse = incumbent_rmse * (1.0 - min_improvement_pct / 100.0)
    if candidate_rmse <= required_rmse:
        return True, f"Candidate RMSE {candidate_rmse:.5f} met threshold {required_rmse:.5f}"
    return False, f"Candidate RMSE {candidate_rmse:.5f} exceeded threshold {required_rmse:.5f}"


def register_model(model_uri: str, model_name: str) -> str:
    """Register a logged model artifact in Unity Catalog and return its version."""
    import mlflow

    mlflow.set_registry_uri("databricks-uc")
    return str(mlflow.register_model(model_uri=model_uri, name=model_name).version)


def deploy_model(
    model_name: str,
    version: str,
    candidate_rmse: float,
    incumbent_rmse: float | None,
    min_improvement_pct: float = 0.0,
) -> DeploymentDecision:
    """Promote a registered version to @champion only when its quality gate passes."""
    from mlflow import MlflowClient

    promote, reason = should_promote(candidate_rmse, incumbent_rmse, min_improvement_pct)
    client = MlflowClient(registry_uri="databricks-uc")
    client.set_model_version_tag(model_name, version, "promotion_gate", "passed" if promote else "failed")
    client.set_model_version_tag(model_name, version, "promotion_reason", reason)
    if promote:
        client.set_registered_model_alias(model_name, "champion", version)
    return DeploymentDecision(promote, version, candidate_rmse, incumbent_rmse, reason)
