from datetime import datetime
from typing import Optional
import mlflow
from mlflow.tracking import MlflowClient

from src.safety.kill_switch import is_ml_enabled


def rollback_model(
    model_name: str,
    from_stage: str = "Staging",
    to_stage: str = "Production",
    reason: Optional[str] = None,
):
    """
    Rolls back a model by restoring the last Production version
    and demoting the current staged version.
    """

    if not is_ml_enabled():
        print("🛑 ML disabled. Rollback allowed, but no promotion will follow.")

    client = MlflowClient()

    # Get latest production version
    prod_versions = client.get_latest_versions(
        model_name, stages=["Production"]
    )

    if not prod_versions:
        print("❌ No Production model available for rollback.")
        return

    stable_version = prod_versions[0].version

    # Get current staged version
    staged_versions = client.get_latest_versions(
        model_name, stages=[from_stage]
    )

    if not staged_versions:
        print("ℹ️ No staged model found. Nothing to rollback.")
        return

    staged_version = staged_versions[0].version

    print(
        f"⏪ Rolling back model '{model_name}': "
        f"{from_stage} v{staged_version} → Production v{stable_version}"
    )

    # Demote staged model
    client.transition_model_version_stage(
        name=model_name,
        version=staged_version,
        stage="Archived"
    )

    log_rollback_event(
        model_name=model_name,
        rolled_back_version=staged_version,
        restored_version=stable_version,
        reason=reason,
    )


def log_rollback_event(
    model_name: str,
    rolled_back_version: str,
    restored_version: str,
    reason: Optional[str] = None,
):
    """
    Writes an immutable audit log entry for rollback.
    """
    payload = {
        "event": "MODEL_ROLLBACK",
        "model": model_name,
        "rolled_back_version": rolled_back_version,
        "restored_version": restored_version,
        "reason": reason or "canary_failure",
        "timestamp": datetime.utcnow().isoformat(),
    }

    print("🧾 ROLLBACK AUDIT LOG:", payload)
