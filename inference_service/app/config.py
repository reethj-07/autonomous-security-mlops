from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # -----------------------------
    # Service metadata
    # -----------------------------
    service_name: str = "security-ml-inference"
    environment: str = Field(default="local")

    # -----------------------------
    # MLflow / Model config
    # -----------------------------
    mlflow_tracking_uri: str = Field(
        default="https://dagshub.com/reethj-07/autonomous-security-mlops.mlflow"
    )

    model_name: str = "security-log-model"
    model_stage: str = Field(default="Staging")

    # -----------------------------
    # Safety controls
    # -----------------------------
    safe_mode: bool = Field(default=False)

    class Config:
        env_prefix = "INFERENCE_"
        env_file = ".env"
        case_sensitive = False


settings = Settings()
