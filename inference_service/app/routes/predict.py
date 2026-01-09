from fastapi import APIRouter, Depends
from time import time

from app.schemas import PredictionRequest, PredictionResponse
from app.model_loader import get_model
from app.safety import enforce_prediction_allowed
from app.auth import require_api_key
from app.config import settings

router = APIRouter(prefix="/predict", tags=["Inference"])


@router.post(
    "",
    response_model=PredictionResponse,
    dependencies=[Depends(require_api_key)],
)
def predict(request: PredictionRequest):
    enforce_prediction_allowed()

    model = get_model()

    features = [[
        request.event_hour,
        request.is_login_failure,
        request.is_privilege_change,
        request.request_length,
        request.has_sql_keywords,
        request.is_admin_path,
    ]]

    start = time()
    prob = float(model.predict(features)[0])
    latency_ms = round((time() - start) * 1000, 2)

    prediction = int(prob >= request.threshold)

    risk = "LOW"
    if prob >= 0.8:
        risk = "CRITICAL"
    elif prob >= 0.5:
        risk = "HIGH"

    return {
        "prediction": prediction,
        "probability": round(prob, 4),
        "risk_level": risk,
        "latency_ms": latency_ms,
        "model_stage": settings.model_stage,
    }
