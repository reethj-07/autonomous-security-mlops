# inference_service/app/main.py

from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi.responses import JSONResponse

from inference_service.app.config import settings
from inference_service.app.rate_limit import limiter
from inference_service.app.middleware import abuse_monitor
from inference_service.app.routes.health import router as health_router
from inference_service.app.routes.predict import router as predict_router


app = FastAPI(
    title="Security ML Inference Service",
    version="1.0.0",
)

# -----------------------------
# Rate Limiting
# -----------------------------
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    lambda r, e: JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded"}
    )
)
app.add_middleware(SlowAPIMiddleware)

# -----------------------------
# Security Middleware
# -----------------------------
app.middleware("http")(abuse_monitor)

# -----------------------------
# Routers
# -----------------------------
app.include_router(health_router)
app.include_router(predict_router)


@app.get("/")
def root():
    return {
        "service": settings.service_name,
        "environment": settings.environment,
        "model_stage": settings.model_stage,
    }
