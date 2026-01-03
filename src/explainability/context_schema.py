from typing import Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass(frozen=True)
class IncidentContext:
    # -----------------------
    # Metadata
    # -----------------------
    incident_id: str
    timestamp: str
    environment: str  # prod / staging / canary

    # -----------------------
    # Model Info
    # -----------------------
    model_name: str
    model_version: str
    model_stage: str

    # -----------------------
    # Trigger Summary
    # -----------------------
    trigger_reason: str
    retrain_triggered: bool

    # -----------------------
    # Evidence
    # -----------------------
    feature_drift: Dict[str, float]
    prediction_drift: Optional[Dict[str, float]]
    anomaly_score_stats: Optional[Dict[str, float]]
    canary_metrics: Optional[Dict[str, float]]

    # -----------------------
    # Safety State
    # -----------------------
    safe_mode: bool
    kill_switch_active: bool

    # -----------------------
    # Human Summary
    # -----------------------
    summary: str

    def to_dict(self) -> Dict:
        return asdict(self)
