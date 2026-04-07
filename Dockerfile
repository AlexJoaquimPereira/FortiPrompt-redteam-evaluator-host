# ============================================================
# FortiPrompt RedTeam Evaluator — Dockerfile
# Base: NVIDIA CUDA 12.1 + PyTorch (fits Tesla T4 free tier)
#
# Build:
#   docker build -t fortiprompt-api .
#
# Run (with GPU):
#   docker run --gpus all -p 8000:8000 \
#     --env-file .env \
#     fortiprompt-api
#
# Run (CPU only, for testing):
#   docker run -p 8000:8000 --env-file .env fortiprompt-api
# ============================================================

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── System deps ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1

WORKDIR /app

# ── Python deps (cached layer) ───────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────
COPY . .

# ── Port & entry point ───────────────────────────────────────
EXPOSE 8000

CMD ["uvicorn", "api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
