import mlflow
from functools import lru_cache
from inference_service.app.config import settings


@lru_cache()
def get_model():
    """
    Load model once per process (safe for FastAPI workers).
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    model_uri = f"models:/{settings.model_name}/{settings.model_stage}"
    return mlflow.pyfunc.load_model(model_uri)