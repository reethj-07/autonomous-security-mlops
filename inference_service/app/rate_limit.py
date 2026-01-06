from slowapi import Limiter
from slowapi.util import get_remote_address
import os

# CI-safe limiter (no .env required)
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[os.getenv("RATE_LIMIT", "100/minute")],
)