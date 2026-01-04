import os

def check_serving_allowed():
    if os.getenv("SAFE_MODE", "false").lower() == "true":
        raise RuntimeError("ML serving disabled (SAFE MODE)")
