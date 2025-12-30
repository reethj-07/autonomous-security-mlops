import os
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "security-log-detector"
RUN_ID_FILE = "artifacts/run_id.txt"


def register_latest_model():
    if not os.path.exists(RUN_ID_FILE):
        raise FileNotFoundError("run_id.txt not found. Training step did not export run_id.")

    with open(RUN_ID_FILE) as f:
        run_id = f.read().strip()

    client = MlflowClient()

    artifacts = client.list_artifacts(run_id)
    artifact_paths = [a.path for a in artifacts]

    print("Available artifacts:", artifact_paths)

    if "model" not in artifact_paths:
        raise RuntimeError(
            f"No model artifact found under run {run_id}. "
            f"Available artifacts: {artifact_paths}"
        )

    model_uri = f"runs:/{run_id}/model"
    print(f"Registering model from {model_uri}")

    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print(f"Registered model version: {result.version}")


if __name__ == "__main__":
    register_latest_model()
