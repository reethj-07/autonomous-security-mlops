import json
import os
from datetime import datetime, timedelta
from typing import Dict
from src.safety.kill_switch import is_ml_enabled


TRIGGER_FILE = "artifacts/retrain_signal.json"
COOLDOWN_HOURS = 24


def _cooldown_passed(last_trigger_time: str) -> bool:
    """
    Prevent retraining storms.
    """
    last_time = datetime.fromisoformat(last_trigger_time)
    return datetime.utcnow() - last_time >= timedelta(hours=COOLDOWN_HOURS)


def emit_retrain_signal(
    decision: Dict[str, object],
    reason: str = "drift_detected"
) -> bool:
    """
    Emits a retraining trigger if allowed by system safety state.
    """

    if not is_ml_enabled():
        print("🛑 ML disabled (SAFE MODE or LOCKDOWN). Retraining blocked.")
        return False

    if not decision.get("retrain", False):
        print("✅ No retraining needed.")
        return False

    if os.path.exists(TRIGGER_FILE):
        with open(TRIGGER_FILE) as f:
            previous = json.load(f)

        if not _cooldown_passed(previous["timestamp"]):
            print("⏳ Cooldown active. Retraining skipped.")
            return False

    os.makedirs(os.path.dirname(TRIGGER_FILE), exist_ok=True)

    payload = {
        "retrain": True,
        "reason": reason,
        "feature_drift": decision.get("feature_drift"),
        "prediction_drift": decision.get("prediction_drift"),
        "timestamp": datetime.utcnow().isoformat()
    }

    with open(TRIGGER_FILE, "w") as f:
        json.dump(payload, f, indent=2)

    print("🚀 Retraining signal emitted.")
    return True

