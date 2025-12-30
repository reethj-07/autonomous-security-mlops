import os
import mlflow
import dagshub
from mlflow.tracking import MlflowClient

# ✅ Ensure we are looking at the right server
dagshub.init(repo_owner='reethj-07', repo_name='autonomous-security-mlops', mlflow=True)

MODEL_NAME = "security-log-detection"
RUN_ID_FILE = "artifacts/run_id.txt"

def register_latest_model():
    # 1. Read the Run ID
    if not os.path.exists(RUN_ID_FILE):
        raise FileNotFoundError("Run ID file not found. Did you run train.py?")
        
    with open(RUN_ID_FILE) as f:
        run_id = f.read().strip()
    
    print(f"Checking run: {run_id}")
    client = MlflowClient()

    # 2. Check for artifacts
    artifacts = client.list_artifacts(run_id)
    artifact_paths = [a.path for a in artifacts]
    print(f"Found artifacts: {artifact_paths}")

    if "model" not in artifact_paths:
        print("❌ ERROR: 'model' folder is missing.")
        print("This usually means the upload failed in train.py.")
        print("Check if 'pip install boto3' is run and internet is active.")
        raise RuntimeError("Model artifact missing.")

    # 3. Register Model
    model_uri = f"runs:/{run_id}/model"
    print(f"Registering model from {model_uri}...")

    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print(f"✅ Success! Registered model version: {result.version}")
    
    # 4. Promote to Staging (Optional but recommended)
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=result.version,
        stage="Staging"
    )
    print(f"🚀 Model promoted to Staging.")

if __name__ == "__main__":
    register_latest_model()