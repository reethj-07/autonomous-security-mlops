# inference_service/app/rate_limit.py

import os
from slowapi import Limiter
from slowapi.util import get_remote_address


def _safe_limiter():
    """
    Create limiter without requiring .env.
    Works in local, CI, Docker, prod.
    """
    try:
        return Limiter(
            key_func=get_remote_address,
            default_limits=[
                os.getenv("RATE_LIMIT", "100/minute")
            ],
        )
    except Exception as e:
        print(f"⚠️ Rate limiter disabled: {e}")
        return None


limiter = _safe_limiter()