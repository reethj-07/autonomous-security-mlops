# =========================
# Stage 1 — Builder
# =========================
FROM python:3.10-slim AS builder

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY inference_service/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# =========================
# Stage 2 — Runtime
# =========================
FROM python:3.10-slim

WORKDIR /app

# Non-root user (security best practice)
RUN useradd -m appuser
USER appuser

# Copy installed Python packages
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy inference service code
COPY inference_service/app ./app
COPY inference_service/requirements.txt ./requirements.txt

# Runtime env
ENV PYTHONPATH=/app
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
