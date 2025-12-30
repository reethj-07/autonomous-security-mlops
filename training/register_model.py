import os
import mlflow

MODEL_NAME = "security-log-detector"

def register_latest_model():
    client = mlflow.tracking.MlflowClient()

    # Get experiment
    experiment = client.get_experiment_by_name("security-log-detection")
    if experiment is None:
        raise ValueError("Experiment not found")

    # Get latest successful run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError("No runs found")

    run = runs[0]
    run_id = run.info.run_id

    model_uri = f"runs:/{run_id}/model"

    print(f"Registering model from: {model_uri}")

    mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print(f"✅ Model registered as '{MODEL_NAME}'")

if __name__ == "__main__":
    register_latest_model()
