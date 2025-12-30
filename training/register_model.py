import mlflow

MODEL_NAME = "security-log-detector"

def register_latest_model():
    with open("artifacts/run_id.txt") as f:
        run_id = f.read().strip()

    model_uri = f"runs:/{run_id}/model"

    print(f"Registering model from: {model_uri}")

    mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print(f"Model registered successfully: {MODEL_NAME}")


if __name__ == "__main__":
    register_latest_model()
