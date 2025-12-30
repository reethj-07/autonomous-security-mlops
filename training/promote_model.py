import mlflow
import os
from mlflow.tracking import MlflowClient

# ✅ CI-FRIENDLY SETUP
# We use os.getenv to read the secrets directly from GitHub Actions
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# Fallback for local testing
if not MLFLOW_TRACKING_URI:
    MLFLOW_TRACKING_URI = "https://dagshub.com/reethj-07/autonomous-security-mlops.mlflow"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_NAME = "security-log-model"

def promote_to_staging():
    client = MlflowClient()
    
    # Get the latest version (that train.py just registered)
    versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
    
    if not versions:
        print(f"❌ No model versions found for '{MODEL_NAME}'. Check DagsHub UI.")
        return

    latest_version = versions[0].version
    print(f"🚀 Found Model Version: {latest_version}")
    
    # Promote it
    print(f"Promoting version {latest_version} to Staging...")
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest_version,
        stage="Staging"
    )
    print("✅ Success! Model is now in Staging.")

if __name__ == "__main__":
    promote_to_staging()