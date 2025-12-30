import mlflow
import dagshub
from mlflow.tracking import MlflowClient

# Connect to DagsHub
dagshub.init(repo_owner='reethj-07', repo_name='autonomous-security-mlops', mlflow=True)

MODEL_NAME = "security-log-model"  # Matches the name in your train.py logs

def promote_to_staging():
    client = MlflowClient()
    
    # Get the latest version that train.py just created
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