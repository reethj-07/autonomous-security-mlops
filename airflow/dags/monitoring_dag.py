# airflow/dags/monitoring_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from airflow.tasks.drift_task import compute_drift
from airflow.tasks.decision_task import decide_retraining
from airflow.tasks.retrain_task import run_retraining

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="security_monitoring_dag",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="*/6 * * * *",  # every 6 hours
    catchup=False,
    tags=["security", "monitoring", "drift"],
) as dag:

    drift_check = PythonOperator(
        task_id="compute_drift_metrics",
        python_callable=compute_drift,
    )

    decision = PythonOperator(
        task_id="decide_retraining",
        python_callable=decide_retraining,
    )

    retrain = PythonOperator(
        task_id="retrain_model",
        python_callable=run_retraining,
    )

    

    drift_check >> decision >> retrain
