# airflow/tasks/decision_task.py
import json
from pathlib import Path

SIGNAL_FILE = Path("airflow/signals/retrain_signal.json")

def decide_retraining(**context):
    drift_score = context["ti"].xcom_pull(
        task_ids="compute_drift_metrics",
        key="drift_score"
    )

    decision = {
        "retrain": bool(drift_score and drift_score > 0.2),
        "drift_score": drift_score
    }

    SIGNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    SIGNAL_FILE.write_text(json.dumps(decision, indent=2))

    print("📡 Retraining decision:", decision)
    return decision["retrain"]
