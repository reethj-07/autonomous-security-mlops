# inference_service/app/metrics.py

from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import APIRouter, Response

router = APIRouter(tags=["Metrics"])

# --------------------------------------
# Model runtime metrics
# --------------------------------------
MODEL_STAGE = Gauge(
    "model_loaded_stage",
    "Currently loaded ML model stage",
    ["stage"],
)


@router.get("/metrics")
def metrics():
    """
    Prometheus-compatible metrics endpoint
    """
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
