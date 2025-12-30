import mlflow

MODEL_NAME = "security-log-detector"


def register_latest_model():
    with open("training/latest_run_id.txt") as f:
        run_id = f.read().strip()

    model_uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print(
        f"Successfully registered model '{MODEL_NAME}' "
        f"(version {result.version}) from run {run_id}"
    )


if __name__ == "__main__":
    register_latest_model()
