# inference_service/app/middleware.py

import time
from fastapi import Request


async def abuse_monitor(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = round((time.time() - start) * 1000, 2)

    if latency > 500:
        # Hook for future alerting
        print(f"⚠️ High latency detected: {latency} ms")

    return response
