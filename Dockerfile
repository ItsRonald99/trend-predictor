# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.lock.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.lock.txt

COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages and the editable install
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code and runtime assets only (no notebooks, reports, raw data)
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/ ./data/processed/

# Re-install the package in the final layer so entry points resolve
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e . --no-deps

EXPOSE 8000

CMD ["uvicorn", "trend_predictor.api:app", "--host", "0.0.0.0", "--port", "8000"]
