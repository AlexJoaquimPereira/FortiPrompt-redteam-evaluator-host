#!/bin/bash
# =============================================================================
# deploy_aws.sh — FortiPrompt API deployment on AWS EC2 / SageMaker notebook
#
# Usage:
#   chmod +x deploy_aws.sh
#   ./deploy_aws.sh
#
# This script:
#   1. Installs system dependencies
#   2. Installs Python packages
#   3. Configures environment variables
#   4. Starts the FastAPI server with Uvicorn
#
# For production, consider running Uvicorn behind nginx and using
# systemd or PM2 for process management.
# =============================================================================

set -e

echo "=== FortiPrompt API — AWS Deployment ==="

# ── 1. System dependencies ────────────────────────────────────────────────
echo "[1/5] Installing system packages..."
sudo apt-get update -q
sudo apt-get install -y -q python3-pip python3-venv git curl

# ── 2. Python environment ─────────────────────────────────────────────────
echo "[2/5] Setting up Python venv..."
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# ── 3. Environment ────────────────────────────────────────────────────────
echo "[3/5] Loading environment variables..."
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
    echo "  .env loaded."
else
    echo "  WARNING: .env not found. Copy .env.example to .env and fill in values."
    echo "  cp .env.example .env && nano .env"
    exit 1
fi

# ── 4. HuggingFace login (needed to download WildGuard) ──────────────────
if [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "[4/5] Logging in to HuggingFace Hub..."
    huggingface-cli login --token "$HUGGING_FACE_HUB_TOKEN"
else
    echo "[4/5] HUGGING_FACE_HUB_TOKEN not set — skipping HF login."
    echo "  WildGuard requires gated access. Run: huggingface-cli login"
fi

# ── 5. Start server ───────────────────────────────────────────────────────
echo "[5/5] Starting Uvicorn..."
HOST="${API_HOST:-0.0.0.0}"
PORT="${API_PORT:-8000}"
WORKERS="${API_WORKERS:-1}"   # 1 worker to share the GPU

echo "  Listening on http://$HOST:$PORT"
echo "  Workers: $WORKERS (set API_WORKERS= to change)"
echo "  Press Ctrl+C to stop."

uvicorn api:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level info
