# inference_service/app/metrics.py

import time
from fastapi import APIRouter, Response
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# --------------------------------------
# Router
# --------------------------------------
router = APIRouter(tags=["Metrics"])

# --------------------------------------
# App lifecycle
# --------------------------------------
START_TIME = time.time()

# --------------------------------------
# Request metrics
# --------------------------------------
REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total number of inference requests",
)

REQUEST_LATENCY = Histogram(
    "inference_request_latency_ms",
    "Inference request latency in milliseconds",
    buckets=(10, 25, 50, 100, 200, 500, 1000, 2000),
)

# --------------------------------------
# Model runtime metrics
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
# Health summary (CI + LB safe)
# --------------------------------------
def get_metrics():
    """
    Lightweight runtime summary for /health and CI
    """
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
