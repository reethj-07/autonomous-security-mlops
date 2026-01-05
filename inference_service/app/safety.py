from fastapi import HTTPException, status
from inference_service.app.config import settings


def enforce_inference_safety():
    """
    Central safety gate for inference.
    """

    if settings.safe_mode:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference disabled: SAFE MODE active"
        )
