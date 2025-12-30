import mlflow
import os
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(
    os.getenv(
        "MLFLOW_TRACKING_URI",
        "https://dagshub.com/reethj-07/autonomous-security-mlops.mlflow"
    )
)

MODEL_NAME = "security-log-detector"


def register_latest_model():
    client = MlflowClient()

    # Get latest run from experiment
    experiment = client.get_experiment_by_name("security-log-detection")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )

    if not runs:
        raise RuntimeError("No runs found to register")

    run = runs[0]
    run_id = run.info.run_id

    model_uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print(f"Registered model version: {result.version}")


if __name__ == "__main__":
    register_latest_model()
