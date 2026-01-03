from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from src.deployment.canary_evaluator import evaluate_canary
from src.deployment.rollback_executor import rollback_model
from src.safety.kill_switch import is_ml_enabled


MODEL_NAME = "security-log-model"

# -----------------------------
# Dummy metric loaders
# (Replace later with real monitoring sinks)
# -----------------------------
def load_baseline_metrics():
    """
    Metrics from last stable Production model
    """
    return {
        "alerts_per_min": 5,
        "prediction_entropy": 0.35,
        "p95_latency_ms": 120,
    }


def load_canary_metrics():
    """
    Metrics from Canary / Staging model
    """
    return {
        "alerts_per_min": 6,
        "prediction_entropy": 0.36,
        "p95_latency_ms": 130,
    }


# -----------------------------
# Canary Orchestration Logic
# -----------------------------
def orchestrate_canary():
    """
    Evaluates canary model health and
    triggers rollback if guardrails fail.
    """

    if not is_ml_enabled():
        print("🛑 ML disabled (SAFE MODE). Canary evaluation skipped.")
        return

    baseline_metrics = load_baseline_metrics()
    canary_metrics = load_canary_metrics()

    result = evaluate_canary(
        canary_metrics=canary_metrics,
        baseline_metrics=baseline_metrics
    )

    print("🐦 Canary evaluation result:", result)

    decision = result.get("decision")

    if decision == "ROLLBACK":
        print("⏪ Canary failed guardrails. Executing rollback.")
        rollback_model(
            model_name=MODEL_NAME,
            reason="canary_guardrail_violation"
        )

    elif decision == "PROMOTE":
        print("🚀 Canary passed. Eligible for production promotion.")
        # Promotion is handled in a separate controlled DAG / CI step

    else:
        print("⏳ Canary inconclusive. Continuing observation window.")


# -----------------------------
# DAG Definition
# -----------------------------
default_args = {
    "owner": "mlops",
    "depends_on_past": False,
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
