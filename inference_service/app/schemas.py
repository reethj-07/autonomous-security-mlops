from pydantic import BaseModel, Field
from typing import Dict, Any


class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(
        ..., description="Numerical feature map for inference"
    )


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str
    model_stage: str
    metadata: Dict[str, Any]
