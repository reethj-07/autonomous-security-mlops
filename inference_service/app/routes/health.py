from fastapi import APIRouter
from app.model_loader import get_model

router = APIRouter(tags=["Health"])


@router.get("/health")
def health():
    try:
        get_model()
        return {
            "status": "ok",
            "model": "loaded",
        }
    except Exception as e:
        return {
            "status": "degraded",
            "model": "unavailable",
            "error": str(e),
        }
