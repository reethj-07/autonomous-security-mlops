# inference_service/app/metrics.py

import time
from fastapi import APIRouter, Response
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST

# --------------------------------------
# Router
# --------------------------------------
router = APIRouter(tags=["Metrics"])

# --------------------------------------
# Runtime tracking
# --------------------------------------
START_TIME = time.time()

# --------------------------------------
# Prometheus Gauges
# --------------------------------------
MODEL_LOADED = Gauge(
    "model_loaded",
    "Whether a model is currently loaded (1 = yes, 0 = no)",
)

MODEL_STAGE = Gauge(
    "model_loaded_stage",
    "Currently loaded ML model stage",
    ["stage"],
)

# --------------------------------------
# Health / CI-friendly metrics
# --------------------------------------
def get_metrics():
    """
    Lightweight runtime metrics used by /health and CI checks
    """
    # Local import to avoid circular dependency
    from app.model_loader import MODEL_STATE

    return {
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "model_loaded": MODEL_STATE["loaded"],
        "served_model_stage": MODEL_STATE["served_stage"],
    }

# --------------------------------------
# Prometheus endpoint
# --------------------------------------
@router.get("/metrics")
def metrics():
    """
    Prometheus-compatible metrics endpoint
    """
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
