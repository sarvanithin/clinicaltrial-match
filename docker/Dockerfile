FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY clinicaltrial_match/ ./clinicaltrial_match/

ARG EXTRAS="mpp"
RUN pip install --no-cache-dir ".[${EXTRAS}]"

ENV CTM_DB__PATH=/data/clinicaltrial_match.db
ENV CTM_API__HOST=0.0.0.0
ENV CTM_API__PORT=8000

RUN mkdir -p /data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

CMD ["python", "-m", "clinicaltrial_match", "serve"]
