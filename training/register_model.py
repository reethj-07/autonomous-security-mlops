import os
import mlflow
from mlflow.tracking import MlflowClient


MODEL_NAME = "security-log-detector"


def register_latest_model():
    client = MlflowClient()

    experiment = client.get_experiment_by_name("security-log-detection")
    if experiment is None:
        raise RuntimeError("Experiment not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1
    )

    if not runs:
        raise RuntimeError("No MLflow runs found")

    run = runs[0]
    run_id = run.info.run_id

    model_uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print(f"Successfully registered model '{MODEL_NAME}' (version {result.version})")


if __name__ == "__main__":
    register_latest_model()
