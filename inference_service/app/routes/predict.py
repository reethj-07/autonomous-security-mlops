from fastapi import APIRouter, Depends
from time import time
import pandas as pd

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
    """
    Run ML inference on a single security event.
    """

    # 🔐 SAFETY GATE
    enforce_prediction_allowed()

    model = get_model()

    # ✅ IMPORTANT: PyFunc expects DataFrame
    input_df = pd.DataFrame([{
        "event_hour": request.event_hour,
        "is_login_failure": request.is_login_failure,
        "is_privilege_change": request.is_privilege_change,
        "request_length": request.request_length,
        "has_sql_keywords": request.has_sql_keywords,
        "is_admin_path": request.is_admin_path,
    }])

    start = time()

    # ✅ PyFunc-compatible prediction
    preds = model.predict(input_df)

    latency_ms = round((time() - start) * 1000, 2)

    # ✅ Handle different return formats safely
    if hasattr(preds, "__len__"):
        prob = float(preds[0])
    else:
        prob = float(preds)

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
