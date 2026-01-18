# inference_service/app/routes/health.py

from fastapi import APIRouter
from app.metrics import get_metrics

router = APIRouter(tags=["Health"])


@router.get("/health")
def health():
    """
    Service health endpoint used by:
    - Load balancers
    - CI smoke tests
    - Production monitoring
    """
    metrics = get_metrics()

    return {
        "status": "ok" if metrics["model_loaded"] else "degraded",
        "uptime_seconds": metrics["uptime_seconds"],
        "model_loaded": metrics["model_loaded"],
        "served_model_stage": metrics["served_model_stage"],
    }
