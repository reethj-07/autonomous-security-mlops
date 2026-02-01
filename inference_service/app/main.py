from fastapi import FastAPI
from fastapi.responses import JSONResponse

from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.config import settings
from app.rate_limit import limiter
from app.middleware import abuse_monitor
from app.routes.health import router as health_router
from app.routes.predict import router as predict_router
from app.routes.metrics import router as metrics_router

app = FastAPI(
    title="Security ML Inference Service",
    version="1.0.0",
)

# -----------------------------
# Rate Limiting (CI + Docker SAFE)
# -----------------------------
if limiter is not None:
    app.state.limiter = limiter

    app.add_exception_handler(
        RateLimitExceeded,
        lambda r, e: JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"},
        ),
    )

    app.add_middleware(SlowAPIMiddleware)
else:
    print("⚠️ Rate limiting disabled (no limiter available)")

# -----------------------------
# Security Middleware
# -----------------------------
app.middleware("http")(abuse_monitor)

# -----------------------------
# Routers
# -----------------------------
app.include_router(health_router)
app.include_router(predict_router)
app.include_router(metrics_router)


# -----------------------------
# Root
# -----------------------------
@app.get("/")
def root():
    return {
        "service": settings.service_name,
        "environment": settings.environment,
        "model_stage": settings.model_stage,
    }
