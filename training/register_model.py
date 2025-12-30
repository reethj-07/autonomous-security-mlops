import os
import mlflow

MODEL_NAME = "security-log-detector"
RUN_ID_FILE = "artifacts/run_id.txt"


def register_latest_model():
    if not os.path.exists(RUN_ID_FILE):
        raise RuntimeError("run_id.txt missing – training step failed")

    with open(RUN_ID_FILE) as f:
        run_id = f.read().strip()

    model_uri = f"runs:/{run_id}/model"
    print(f"Registering model from: {model_uri}")

    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print(f"Model registered as version {result.version}")


if __name__ == "__main__":
    register_latest_model()
