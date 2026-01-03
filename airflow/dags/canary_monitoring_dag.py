from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from src.deployment.canary_evaluator import evaluate_canary
from src.safety.kill_switch import is_ml_enabled


# -----------------------------
# Dummy metric loaders (replace later)
# -----------------------------
def load_baseline_metrics():
    return {
        "alerts_per_min": 5,
        "prediction_entropy": 0.35,
        "p95_latency_ms": 120,
    }


def load_canary_metrics():
    return {
        "alerts_per_min": 6,
        "prediction_entropy": 0.36,
        "p95_latency_ms": 130,
    }


def orchestrate_canary():
    if not is_ml_enabled():
        print("🛑 ML disabled. Canary evaluation skipped.")
        return

    baseline = load_baseline_metrics()
    canary = load_canary_metrics()

    result = evaluate_canary(canary, baseline)

    print("🐦 Canary decision:", result)

    if result["decision"] == "ROLLBACK":
        print("⏪ Rolling back canary model")
        # Placeholder: rollback logic (MLflow stage revert)

    elif result["decision"] == "PROMOTE":
        print("🚀 Canary passed. Eligible for next stage")

    else:
        print("⏳ Canary inconclusive. Extending evaluation window")


default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="canary_monitoring_dag",
    start_date=datetime(2025, 1, 1),
    schedule_interval="*/30 * * * *",  # every 30 minutes
    catchup=False,
    default_args=default_args,
    tags=["canary", "safety", "deployment"],
) as dag:

    canary_check = PythonOperator(
        task_id="evaluate_canary_model",
        python_callable=orchestrate_canary,
    )
