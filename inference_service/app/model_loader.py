import os
import mlflow
from functools import lru_cache
from app.config import settings


@lru_cache()
def get_model():
    """
    Load MLflow model with safe fallback:
    Production → Staging → FAIL
    """

    model_name = settings.model_name
    requested_stage = settings.model_stage

    # Ensure MLflow auth works everywhere
    if settings.mlflow_tracking_username:
        os.environ["MLFLOW_TRACKING_USERNAME"] = settings.mlflow_tracking_username
    if settings.mlflow_tracking_password:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.mlflow_tracking_password

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    # --------------------------------------------------
    # 1️⃣ Try requested stage
    # --------------------------------------------------
    try:
        print(f"🔍 Loading model: {model_name} [{requested_stage}]")
        return mlflow.pyfunc.load_model(
            f"models:/{model_name}/{requested_stage}"
        )
    except Exception as e:
        print(f"⚠️ Failed loading {requested_stage}: {e}")

    # --------------------------------------------------
    # 2️⃣ Safe fallback → Staging
    # --------------------------------------------------
    if requested_stage.lower() != "staging":
        try:
            print("🔁 Falling back to STAGING")
            return mlflow.pyfunc.load_model(
                f"models:/{model_name}/Staging"
            )
        except Exception as e:
            print(f"❌ Staging fallback failed: {e}")

    # --------------------------------------------------
    # 3️⃣ Hard fail (correct behavior)
    # --------------------------------------------------
    raise RuntimeError(
        f"❌ No usable model found for {model_name} "
        f"(requested={requested_stage})"
    )
