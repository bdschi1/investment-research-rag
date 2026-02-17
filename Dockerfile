# ==========================================================================
# Multi-stage build for AWS Lambda container deployment
# ==========================================================================

# Stage 1: Builder — install dependencies with build tools
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY cli/ cli/

RUN pip install --no-cache-dir --target /build/deps ".[aws,faiss]"

# Stage 2: Runtime — minimal image
FROM python:3.11-slim

RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

COPY --from=builder /build/deps /app/deps
COPY --from=builder /build/src /app/src
COPY --from=builder /build/cli /app/cli
COPY lambda/ lambda/
COPY settings.yaml .

ENV PYTHONPATH="/app/deps:/app/src:/app"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir awslambdaric

USER appuser

ENTRYPOINT ["python", "-m", "awslambdaric"]
CMD ["lambda.query_handler.handler"]
