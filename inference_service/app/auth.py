# inference_service/app/auth.py

from fastapi import Header, HTTPException, status
from inference_service.app.config import settings


def require_api_key(x_api_key: str = Header(...)):
    if settings.api_key is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key not configured"
        )

    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
