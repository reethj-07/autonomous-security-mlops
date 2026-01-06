from fastapi import HTTPException, status

from app.config import settings


def enforce_prediction_allowed():
    """
    Blocks inference when SAFE MODE is enabled.
    """
    if settings.safe_mode:
        raise HTTPException (
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SAFE MODE enabled: inference temporarily disabled",
            )


def enforce_inference_safety():
    """
    Central safety gate for inference.
    """

    if settings.safe_mode:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference disabled: SAFE MODE active"

        )