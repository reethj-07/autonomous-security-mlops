# inference_service/app/routes/health.py

from fastapi import APIRouter
from app.model_loader import MODEL_STATE
from app.metrics import get_metrics

router = APIRouter(tags=["Health"])


@router.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_STATE["loaded"],
        "served_stage": MODEL_STATE["served_stage"],
        "metrics": get_metrics(),
    }
