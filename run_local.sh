#!/bin/bash
# Local development runner — runs the app outside Docker for faster iteration.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

set -a
source "$SCRIPT_DIR/.env"
set +a

# Share the Docker model cache so we don't re-download gigabytes of pyannote/whisper.
export HF_HOME="$SCRIPT_DIR/volumes/huggingface_cache"
export WHISPER_CACHE_DIR="$SCRIPT_DIR/volumes/whisper_cache"

export DATA_PATH="$SCRIPT_DIR/data"
export VOLUMES_PATH="$SCRIPT_DIR/volumes"
export DATABASE_URL="sqlite:///$SCRIPT_DIR/volumes/speakers.db"

source "$SCRIPT_DIR/venv/bin/activate"

# faster-whisper/ctranslate2 need cuDNN + cuBLAS on LD_LIBRARY_PATH.
# Glob the site-packages path so the script doesn't care whether the venv
# was created with python3.11, 3.12, or a future interpreter.
SITE_PACKAGES=$(echo "$SCRIPT_DIR"/venv/lib/python*/site-packages)
CUDNN_PATH="$SITE_PACKAGES/nvidia/cudnn/lib"
CUBLAS_PATH="$SITE_PACKAGES/nvidia/cublas/lib"
export LD_LIBRARY_PATH="$CUDNN_PATH:$CUBLAS_PATH:${LD_LIBRARY_PATH:-}"
echo "cuDNN/cuBLAS libraries: $CUDNN_PATH"

mkdir -p volumes data/recordings data/temp

export PORT="${PORT:-8418}"
echo "Starting speaker diarization app locally..."
echo "API:      http://localhost:$PORT"
echo "API Docs: http://localhost:$PORT/docs"
echo ""

exec python -m app.main
