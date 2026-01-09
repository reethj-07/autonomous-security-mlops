# inference_service/app/config.py

import os
from pydantic_settings import BaseSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


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
    # Security
    # -----------------------------
    safe_mode: bool = False
    api_key: str | None = None
    rate_limit: str = "30/minute"

    # ✅ Pydantic v2 config (ONLY THIS)
    model_config = SettingsConfigDict(
        env_prefix="INFERENCE_",
        env_file=".env",
        case_sensitive=False,
        protected_namespaces=(),  # fixes model_* warnings
    )


settings = Settings()

# -----------------------------
# Export MLflow creds to env
# -----------------------------
if settings.mlflow_tracking_username:
    os.environ["MLFLOW_TRACKING_USERNAME"] = settings.mlflow_tracking_username

if settings.mlflow_tracking_password:
    os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.mlflow_tracking_password
