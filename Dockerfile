# Lightweight Python base; CUDA libraries are pip-installed below.
FROM python:3.11-slim

WORKDIR /app

# System deps:
#   ffmpeg          — audio decoding (pydub + faster-whisper)
#   git             — some pip deps pull from git refs during install
#   build-essential — native extension compilation
#   tini            — PID 1 signal forwarding so SIGTERM reaches uvicorn and
#                     pyannote/faster-whisper get a chance to release VRAM
#   curl            — used by the healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    tini \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# 1. Install PyTorch with CUDA support FIRST
# Using CUDA 12.8 wheels for Blackwell (RTX 5090 sm_120) support.
# PyTorch 2.10.x includes sm_120 arch support needed for RTX 5090.
RUN pip install torch==2.10.0 torchaudio==2.10.0 torchcodec==0.10.0 --index-url https://download.pytorch.org/whl/cu128

# 2. CUDA libraries required by ctranslate2 (faster-whisper) and torchcodec
#    since we aren't using the nvidia/cuda base image.
RUN pip install nvidia-cudnn-cu12==9.* nvidia-cublas-cu12 nvidia-npp-cu12

# 3. Remaining deps, pinned against the torch* wheels we just installed.
COPY requirements.txt .
RUN echo "torch==2.10.0" > /tmp/constraints.txt && \
    echo "torchaudio==2.10.0" >> /tmp/constraints.txt && \
    echo "torchcodec==0.10.0" >> /tmp/constraints.txt && \
    pip install --no-cache-dir -c /tmp/constraints.txt -r requirements.txt

# Copy application code
COPY app/ /app/app/

# Create a non-root user. UID 1000 so host-mounted volumes with default user
# ownership stay writable. All state lives under /app, which is chown'd below.
RUN groupadd --system --gid 1000 app \
 && useradd --system --uid 1000 --gid app --home-dir /app --shell /usr/sbin/nologin app \
 && mkdir -p /app/data /app/volumes /app/backups /app/certs /app/volumes/huggingface_cache \
 && chown -R app:app /app

# Expose port
EXPOSE 8418

# Runtime environment:
#   PYTHONUNBUFFERED=1 — log lines flush immediately
#   DATA_PATH/VOLUMES_PATH — consumed by the app for recordings and DB
#   HF_HOME lives under /app/volumes so it's writable by the non-root user and
#           persists through the existing ./volumes host mount.
ENV PYTHONUNBUFFERED=1
ENV DATA_PATH=/app/data
ENV VOLUMES_PATH=/app/volumes
ENV HF_HOME=/app/volumes/huggingface_cache

# LD_LIBRARY_PATH for pip-installed NVIDIA libs (ctranslate2/faster-whisper
# needs cuDNN/cuBLAS; torchcodec needs nvidia-npp).
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/npp/lib:$LD_LIBRARY_PATH

# Drop to the unprivileged user for runtime.
USER app

# start-period is generous — first boot downloads pyannote + whisper models
# which can take several minutes on a fresh volume.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -fsk https://localhost:8418/ >/dev/null 2>&1 \
     || curl -fs http://localhost:8418/ >/dev/null 2>&1 \
     || exit 1

# tini as PID 1 forwards signals correctly
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "app.main"]
