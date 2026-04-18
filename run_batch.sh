#!/bin/bash
# Launcher for multi-GPU batch processing.
# Example: ./run_batch.sh /mnt/data/otherdatas/audio/ --gpus 0,1,2
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

set -a
source "$SCRIPT_DIR/.env"
set +a

source "$SCRIPT_DIR/venv/bin/activate"

export HF_HOME="$SCRIPT_DIR/volumes/huggingface_cache"
export WHISPER_CACHE_DIR="$SCRIPT_DIR/volumes/whisper_cache"
export DATA_PATH="$SCRIPT_DIR/data"
export VOLUMES_PATH="$SCRIPT_DIR/volumes"
export DATABASE_URL="sqlite:///$SCRIPT_DIR/volumes/speakers.db"

# Glob the interpreter-specific site-packages so this script isn't pinned
# to a particular Python version.
SITE_PACKAGES=$(echo "$SCRIPT_DIR"/venv/lib/python*/site-packages)
CUDNN_PATH="$SITE_PACKAGES/nvidia/cudnn/lib"
CUBLAS_PATH="$SITE_PACKAGES/nvidia/cublas/lib"
export LD_LIBRARY_PATH="$CUDNN_PATH:$CUBLAS_PATH:${LD_LIBRARY_PATH:-}"
echo "cuDNN/cuBLAS: $CUDNN_PATH"
echo ""

exec python "$SCRIPT_DIR/batch_process.py" "$@"
