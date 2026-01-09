# inference_service/app/config.py

import os
from pydantic_settings import BaseSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "security-ml-inference"
    environment: str = "local"

    mlflow_tracking_uri: str = Field(
        default="https://dagshub.com/reethj-07/autonomous-security-mlops.mlflow"
    )
    mlflow_tracking_username: str | None = None
    mlflow_tracking_password: str | None = None

    model_name: str = "security-log-model"
    model_stage: str = "Staging"

    safe_mode: bool = False
    api_key: str | None = None
    rate_limit: str = "30/minute"

    model_config = {
        "env_prefix": "INFERENCE_",
        "case_sensitive": False,
        "protected_namespaces": ()
    }


settings = Settings()

# -----------------------------
# Export MLflow creds to env
# -----------------------------
if settings.mlflow_tracking_username:
    os.environ["MLFLOW_TRACKING_USERNAME"] = settings.mlflow_tracking_username

if settings.mlflow_tracking_password:
    os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.mlflow_tracking_password
