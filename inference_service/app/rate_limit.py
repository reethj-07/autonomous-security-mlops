# inference_service/app/rate_limit.py

import os
from slowapi import Limiter
from slowapi.util import get_remote_address

# ✅ CI-safe: no .env, only env vars
RATE_LIMIT = os.getenv("RATE_LIMIT", "10/minute")

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[RATE_LIMIT],
    enabled=True,
)
