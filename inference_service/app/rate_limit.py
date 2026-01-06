import os
from slowapi import Limiter
from slowapi.util import get_remote_address

# ✅ Read from environment variables ONLY
RATE_LIMIT = os.getenv("RATE_LIMIT", "10/minute")

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[RATE_LIMIT],
)
