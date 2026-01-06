# inference_service/app/config.py

import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # -----------------------------
    # Service metadata
    # -----------------------------
    service_name: str = "security-ml-inference"
    environment: str = "local"

    # -----------------------------
    # MLflow configuration
    # -----------------------------
    mlflow_tracking_uri: str = Field(
        default="https://dagshub.com/reethj-07/autonomous-security-mlops.mlflow"
    )
    mlflow_tracking_username: str | None = None
    mlflow_tracking_password: str | None = None

    model_name: str = "security-log-model"
    model_stage: str = "Staging"

    # -----------------------------
    # Security & Safety Controls
    # -----------------------------
    safe_mode: bool = False

    # API Security
    api_key: str | None = None          # X-API-Key
    rate_limit: str = "30/minute"       # SlowAPI limit

    class Config:
        env_prefix = "INFERENCE_"
        env_file = ".env"
        case_sensitive = False


settings = Settings()


# -----------------------------
# Ensure MLflow auth works everywhere
# -----------------------------
if settings.mlflow_tracking_username:
    os.environ["MLFLOW_TRACKING_USERNAME"] = settings.mlflow_tracking_username

if settings.mlflow_tracking_password:
    os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.mlflow_tracking_password

# ✅ Ensure MLflow auth works in CI & prod
if settings.mlflow_tracking_username:
    os.environ["MLFLOW_TRACKING_USERNAME"] = settings.mlflow_tracking_username
if settings.mlflow_tracking_password:
    os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.mlflow_tracking_password

