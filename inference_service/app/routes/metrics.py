from prometheus_client import Counter, Histogram, Gauge

# Total prediction requests
REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total number of inference requests"
)

# Prediction errors
ERROR_COUNT = Counter(
    "inference_errors_total",
    "Total number of inference errors"
)

# Latency histogram (ms)
LATENCY_HISTOGRAM = Histogram(
    "inference_latency_ms",
    "Inference latency in milliseconds",
    buckets=(10, 25, 50, 100, 250, 500, 1000, 2000)
)

# Loaded model stage (0/1 gauge)
MODEL_STAGE = Gauge(
    "model_loaded_stage",
    "Loaded model stage (1 = active)",
    ["stage"]
)
