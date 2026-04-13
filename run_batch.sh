#!/bin/bash

# Launcher for multi-GPU batch processing
# Usage: ./run_batch.sh /mnt/data/otherdatas/audio/ --gpus 0,1,2

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Load environment
set -a
source "$SCRIPT_DIR/.env"
set +a

# Activate venv
source "$SCRIPT_DIR/venv/bin/activate"

# Set model cache paths
export HF_HOME="$SCRIPT_DIR/volumes/huggingface_cache"
export WHISPER_CACHE_DIR="$SCRIPT_DIR/volumes/whisper_cache"
export DATA_PATH="$SCRIPT_DIR/data"
export VOLUMES_PATH="$SCRIPT_DIR/volumes"
export DATABASE_URL="sqlite:///$SCRIPT_DIR/volumes/speakers.db"

# Set LD_LIBRARY_PATH for cuDNN/cuBLAS (required by faster-whisper/ctranslate2)
CUDNN_PATH="$SCRIPT_DIR/venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
CUBLAS_PATH="$SCRIPT_DIR/venv/lib/python3.12/site-packages/nvidia/cublas/lib"
export LD_LIBRARY_PATH="$CUDNN_PATH:$CUBLAS_PATH:$LD_LIBRARY_PATH"

echo "cuDNN/cuBLAS: $CUDNN_PATH"
echo ""

python "$SCRIPT_DIR/batch_process.py" "$@"
