# inference_service/app/model_loader.py

import mlflow
from functools import lru_cache
from app.config import settings

# Global runtime state (used by /health)
MODEL_STATE = {
    "loaded": False,
    "served_stage": None,
}


@lru_cache()
def get_model():
    """
    Load MLflow model with safe fallback:
    Production → Staging → FAIL
    """

    model_name = settings.model_name
    requested_stage = settings.model_stage

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    # -----------------------------
    # 1️⃣ Try requested stage
    # -----------------------------
    try:
        print(f"🔍 Loading model: {model_name} [{requested_stage}]")
        model = mlflow.pyfunc.load_model(
            f"models:/{model_name}/{requested_stage}"
        )

        MODEL_STATE["loaded"] = True
        MODEL_STATE["served_stage"] = requested_stage
        return model

    except Exception as primary_error:
        print(f"⚠️ Failed loading {requested_stage}: {primary_error}")

    # -----------------------------
    # 2️⃣ Safe fallback → Staging
    # -----------------------------
    if requested_stage.lower() != "staging":
        try:
            print("🔁 Falling back to STAGING")
            model = mlflow.pyfunc.load_model(
                f"models:/{model_name}/Staging"
            )

            MODEL_STATE["loaded"] = True
            MODEL_STATE["served_stage"] = "Staging"
            return model

        except Exception as fallback_error:
            print(f"❌ Staging fallback failed: {fallback_error}")

    # -----------------------------
    # 3️⃣ Hard fail
    # -----------------------------
    MODEL_STATE["loaded"] = False
    MODEL_STATE["served_stage"] = None

    raise RuntimeError(
        f"❌ No usable model found for {model_name} "
        f"(requested={requested_stage})"
    )
