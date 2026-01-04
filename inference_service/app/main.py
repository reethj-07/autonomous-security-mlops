from fastapi import FastAPI
from inference_service.app.config import settings
from inference_service.app.routes.health import router as health_router


app = FastAPI(
    title="Security ML Inference Service",
    version="1.0.0",
)

app.include_router(health_router)


@app.get("/")
def root():
    return {
        "service": settings.service_name,
        "environment": settings.environment,
        "model_stage": settings.model_stage,
    }
