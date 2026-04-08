# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Metadata ──────────────────────────────────────────────────────────────────
LABEL org.opencontainers.image.title="KAVACH-X"
LABEL org.opencontainers.image.description="Multi-Domain Fraud Intelligence Environment — OpenEnv Hackathon"
LABEL org.opencontainers.image.version="2.0.0"

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /kavach-x

# ── System deps (minimal) ─────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps (cached layer) ────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ────────────────────────────────────────────────────────
COPY . .

# ── Environment variables ─────────────────────────────────────────────────────
ENV PYTHONPATH=/kavach-x
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# HF Spaces injects these at runtime — defaults shown for local testing
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
ENV HF_TOKEN="dummy"

# ── Expose port (HF Spaces default) ──────────────────────────────────────────
EXPOSE 7860

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Start FastAPI server ──────────────────────────────────────────────────────
# Starts FastAPI server — handles /reset, /step, /state for OpenEnv validation.
# To run inference baseline: docker run kavach-x python inference.py
CMD ["python", "app.py"]
