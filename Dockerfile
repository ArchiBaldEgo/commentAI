### Builder stage: prefetch wheels (no compilers in final image)
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1
WORKDIR /wheels

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip download -r requirements.txt -d /wheels


### Runtime stage: slim image without build-essential
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy wheels and project metadata
COPY --from=builder /wheels /wheels
COPY requirements.txt ./
COPY pyproject.toml ./

# Install deps from predownloaded wheels, then install project in editable mode
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-index --find-links=/wheels -r requirements.txt

# Copy source and install package
COPY . .
RUN python -m pip install -e .

EXPOSE 8000

ENV MODEL_DIR=/app/models/online \
        UVICORN_WORKERS=2 \
        UVICORN_TIMEOUT=60

VOLUME ["/app/models"]

# Healthcheck without curl/wget
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
u=urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2); \
sys.exit(0 if u.status==200 else 1)" || exit 1

# Use shell to expand environment variables for workers and timeouts
SHELL ["/bin/sh", "-c"]

# Run uvicorn with multiple workers using the module-level app
CMD python -m uvicorn src.sentiment.server:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-2} --timeout-keep-alive ${UVICORN_TIMEOUT:-60}
