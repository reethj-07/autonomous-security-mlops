from fastapi import HTTPException, status
from inference_service.app.config import settings


def is_prediction_allowed() -> bool:
    """
    Returns True if inference is allowed.
    """
    return not settings.safe_mode


def check_safe_mode() -> None:
    """
    Raises HTTP 503 if SAFE MODE is enabled.
    """
    if not is_prediction_allowed():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SAFE MODE enabled. Predictions are temporarily disabled."
        )
