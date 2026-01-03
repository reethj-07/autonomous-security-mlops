import uuid
from datetime import datetime
from typing import Dict, Optional

from src.explainability.context_schema import IncidentContext
from src.safety.kill_switch import is_ml_enabled


def build_incident_context(
    *,
    environment: str,
    model_name: str,
    model_version: str,
    model_stage: str,
    trigger_reason: str,
    retrain_triggered: bool,
    feature_drift: Dict[str, float],
    prediction_drift: Optional[Dict[str, float]] = None,
    anomaly_score_stats: Optional[Dict[str, float]] = None,
    canary_metrics: Optional[Dict[str, float]] = None,
) -> IncidentContext:
    """
    Builds an immutable incident context object.
    """

    safe_mode = not is_ml_enabled()

    summary_parts = []

    if feature_drift:
        summary_parts.append(
            f"Feature drift detected in {len(feature_drift)} features."
        )

    if prediction_drift:
        summary_parts.append("Prediction confidence degradation observed.")

    if anomaly_score_stats:
        summary_parts.append("Elevated anomaly scores detected.")

    if canary_metrics:
        summary_parts.append("Canary metrics evaluated.")

    if safe_mode:
        summary_parts.append("System in SAFE MODE.")

    summary = " ".join(summary_parts) or "No abnormal behavior detected."

    return IncidentContext(
        incident_id=str(uuid.uuid4()),
        timestamp=datetime.utcnow().isoformat(),
        environment=environment,
        model_name=model_name,
        model_version=model_version,
        model_stage=model_stage,
        trigger_reason=trigger_reason,
        retrain_triggered=retrain_triggered,
        feature_drift=feature_drift,
        prediction_drift=prediction_drift,
        anomaly_score_stats=anomaly_score_stats,
        canary_metrics=canary_metrics,
        safe_mode=safe_mode,
        kill_switch_active=safe_mode,
        summary=summary,
    )
