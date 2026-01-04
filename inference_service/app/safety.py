from fastapi import HTTPException
from inference_service.app.config import settings


def check_safe_mode():
    """
    SAFE MODE blocks inference.
    """
    if settings.safe_mode:
        raise HTTPException(
            status_code=503,
            detail="SAFE MODE enabled. Predictions are temporarily disabled."
        )
