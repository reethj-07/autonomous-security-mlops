import time
from fastapi import Request


async def abuse_monitor(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency_ms = round((time.time() - start) * 1000, 2)

    if latency_ms > 500:
        print(f"⚠️ High latency detected: {latency_ms} ms")

    return response
