# inference_service/app/model_loader.py

import mlflow
from functools import lru_cache

from app.config import settings
from app.metrics import MODEL_STAGE

# -------------------------------------------------
# Global runtime state (used by /health)
# -------------------------------------------------
MODEL_STATE = {
    "loaded": False,
    "served_stage": None,
}


@lru_cache()
def get_model():
    """
    Load MLflow model with safe fallback:
    Production → Staging → FAIL

    This function is cached to avoid repeated loads.
    """

    model_name = settings.model_name
    requested_stage = settings.model_stage

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    # Reset metrics (important for reloads / restarts)
    MODEL_STAGE.clear()

    # -------------------------------------------------
    # 1️⃣ Try requested stage
    # -------------------------------------------------
    try:
        print(f"🔍 Loading model: {model_name} [{requested_stage}]")

        model = mlflow.pyfunc.load_model(
            f"models:/{model_name}/{requested_stage}"
        )

        # Runtime state
        MODEL_STATE["loaded"] = True
        MODEL_STATE["served_stage"] = requested_stage

        # Metrics
        MODEL_STAGE.labels(stage=requested_stage).set(1)

        print(f"✅ Model loaded: {model_name} [{requested_stage}]")
        return model

    except Exception as primary_error:
        print(f"⚠️ Failed loading {requested_stage}: {primary_error}")

    # -------------------------------------------------
    # 2️⃣ Safe fallback → Staging
    # -------------------------------------------------
    if requested_stage.lower() != "staging":
        try:
            print("🔁 Falling back to STAGING")

            model = mlflow.pyfunc.load_model(
                f"models:/{model_name}/Staging"
            )

            # Runtime state
            MODEL_STATE["loaded"] = True
            MODEL_STATE["served_stage"] = "Staging"

            # Metrics
            MODEL_STAGE.labels(stage="Staging").set(1)

            print(f"✅ Fallback model loaded: {model_name} [Staging]")
            return model

        except Exception as fallback_error:
            print(f"❌ Staging fallback failed: {fallback_error}")

    # -------------------------------------------------
    # 3️⃣ Hard fail (correct behavior)
    # -------------------------------------------------
    MODEL_STATE["loaded"] = False
    MODEL_STATE["served_stage"] = None

    raise RuntimeError(
        f"❌ No usable model found for {model_name} "
        f"(requested={requested_stage})"
    )
