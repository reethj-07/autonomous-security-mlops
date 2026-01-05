from pydantic import BaseModel, Field, ConfigDict

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_hour: int = Field(..., ge=0, le=23)
    is_login_failure: int = Field(..., ge=0, le=1)
    is_privilege_change: int = Field(..., ge=0, le=1)
    request_length: int = Field(..., ge=0, le=10_000)
    has_sql_keywords: int = Field(..., ge=0, le=1)
    is_admin_path: int = Field(..., ge=0, le=1)

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    latency_ms: float
    model_stage: str