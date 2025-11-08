# syntax=docker/dockerfile:1
# Lightweight, production-ready Python image
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_DISABLE_PIP_VERSION_CHECK=1 \
	PIP_NO_CACHE_DIR=1

# Install required system libs (OpenMP for xgboost/lightgbm, curl for healthcheck)
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
	   libgomp1 \
	   curl \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Create non-root user
RUN useradd -m appuser \
	&& chown -R appuser:appuser /app
USER appuser

# Expose API port (as defined in app settings)
EXPOSE 3001

# Default command: run uvicorn ASGI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3001", "--log-level", "info"]

