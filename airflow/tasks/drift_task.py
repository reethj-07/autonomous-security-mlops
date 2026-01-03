# airflow/tasks/drift_task.py
def compute_drift(**context):
    """
    Computes drift score and pushes it to XCom.
    """
    # Placeholder example (replace with real drift logic)
    drift_score = 0.25

    context["ti"].xcom_push(
        key="drift_score",
        value=drift_score
    )

    print(f"📊 Drift score computed: {drift_score}")
    return drift_score
