#!/bin/bash

# Local development runner for speaker diarization app
# This script runs the app outside of Docker for faster development

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Load environment variables
set -a
source .env
set +a

# Use the same model cache as Docker to avoid re-downloading
export HF_HOME="$SCRIPT_DIR/volumes/huggingface_cache"
export WHISPER_CACHE_DIR="$SCRIPT_DIR/volumes/whisper_cache"

# Set data paths for local development (not Docker /app paths)
export DATA_PATH="$SCRIPT_DIR/data"
export VOLUMES_PATH="$SCRIPT_DIR/volumes"
export DATABASE_URL="sqlite:///$SCRIPT_DIR/volumes/speakers.db"

# Activate virtual environment
source venv/bin/activate

# Set LD_LIBRARY_PATH for cuDNN 9 and cuBLAS (required by faster-whisper/ctranslate2)
CUDNN_PATH="$SCRIPT_DIR/venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
CUBLAS_PATH="$SCRIPT_DIR/venv/lib/python3.12/site-packages/nvidia/cublas/lib"
export LD_LIBRARY_PATH="$CUDNN_PATH:$CUBLAS_PATH:$LD_LIBRARY_PATH"
echo "cuDNN/cuBLAS libraries: $CUDNN_PATH"

# Create necessary directories
mkdir -p volumes data/recordings data/temp

# Run the application
export PORT=${PORT:-8418}
echo "Starting speaker diarization app locally..."
echo "Docs: http://localhost:$PORT/docs"
echo "API: http://localhost:$PORT"
echo "API Docs: http://localhost:$PORT/docs"
echo ""

python -m app.main
