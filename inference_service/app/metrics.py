# inference_service/app/metrics.py

from threading import Lock

_METRICS = {
    "total_requests": 0,
    "failed_requests": 0,
}

_lock = Lock()


def record_request(success: bool):
    with _lock:
        _METRICS["total_requests"] += 1
        if not success:
            _METRICS["failed_requests"] += 1


def get_metrics():
    return dict(_METRICS)
