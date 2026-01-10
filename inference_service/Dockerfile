# =========================
# Stage 1 — Builder
# =========================
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY inference_service/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# =========================
# Stage 2 — Runtime
# =========================
FROM python:3.10-slim

WORKDIR /app

LABEL org.opencontainers.image.source="https://github.com/reethj-07/autonomous-security-mlops"
LABEL org.opencontainers.image.description="Security ML Inference Service"
LABEL org.opencontainers.image.version="latest"

RUN useradd -m appuser
USER appuser

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY inference_service/app ./app
COPY inference_service/requirements.txt ./requirements.txt

ENV PYTHONPATH=/app
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
