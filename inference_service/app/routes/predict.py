from fastapi import APIRouter
from time import time

from inference_service.app.schemas import PredictionRequest, PredictionResponse
from inference_service.app.model_loader import get_model
from inference_service.app.config import settings
from inference_service.app.safety import check_safe_mode

router = APIRouter(prefix="/predict", tags=["Inference"])


@router.post("", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Run ML inference on a single security event.
    """

    # -----------------------------
    # SAFETY GATE
    # -----------------------------
    check_safe_mode()

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
    prob = float(model.predict_proba(features)[0][1])
    latency_ms = round((time() - start) * 1000, 2)

    prediction = int(prob >= request.threshold)

    if prob >= 0.8:
        risk = "CRITICAL"
    elif prob >= 0.5:
        risk = "HIGH"
    else:
        risk = "LOW"

    return {
        "prediction": prediction,
        "probability": round(prob, 4),
        "risk_level": risk,
        "latency_ms": latency_ms,
        "model_stage": settings.model_stage,
    }
