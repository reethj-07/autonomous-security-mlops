import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_NAME = "security-log-detector"
MIN_F1 = 0.60


def promote_if_valid():
    client = MlflowClient()

    # Get latest model version
    versions = client.get_latest_versions(
        MODEL_NAME,
        stages=["None"]
    )

    if not versions:
        print("No candidate models to promote")
        return

    version = versions[0]
    run_id = version.run_id

    run = client.get_run(run_id)
    f1 = run.data.metrics.get("f1", 0)

    print(f"Evaluating model v{version.version} with F1={f1}")

    if f1 >= MIN_F1:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version.version,
            stage="Staging",
            archive_existing_versions=False
        )
        print("✅ Model promoted to STAGING")
    else:
        print("❌ Model blocked due to insufficient F1 score")


if __name__ == "__main__":
    promote_if_valid()
