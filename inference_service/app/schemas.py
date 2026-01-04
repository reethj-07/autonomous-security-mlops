from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    event_hour: int = Field(..., ge=0, le=23)
    is_login_failure: int
    is_privilege_change: int
    request_length: int
    has_sql_keywords: int
    is_admin_path: int

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    latency_ms: float
    model_stage: str
