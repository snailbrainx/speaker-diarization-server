# Use a lightweight Python base image instead of the heavy PyTorch one
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# ffmpeg: for audio processing
# git: required by some Python deps during install
# build-essential: for compiling native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# 1. Install PyTorch with CUDA support FIRST
# This avoids downloading the huge default CPU wheels or reinstalling later
# Using CUDA 12.8 wheels for Blackwell (RTX 5090 sm_120) support
# PyTorch 2.10.x includes sm_120 arch support needed for RTX 5090
RUN pip install torch==2.10.0 torchaudio==2.10.0 torchcodec==0.10.0 --index-url https://download.pytorch.org/whl/cu128

# 2. Install CUDA libraries for faster-whisper / ctranslate2 and torchcodec
# These are needed because we aren't using the nvidia/cuda base image
# - nvidia-npp-cu12: Required by torchcodec for audio decoding
RUN pip install nvidia-cudnn-cu12==9.* nvidia-cublas-cu12 nvidia-npp-cu12

# 3. Install remaining dependencies
# Create constraints file to prevent torch packages from being reinstalled by other deps
COPY requirements.txt .
RUN echo "torch==2.10.0" > /tmp/constraints.txt && \
    echo "torchaudio==2.10.0" >> /tmp/constraints.txt && \
    echo "torchcodec==0.10.0" >> /tmp/constraints.txt && \
    pip install --no-cache-dir -c /tmp/constraints.txt -r requirements.txt

# Copy application code
COPY app/ /app/app/

# Create directories for data persistence
RUN mkdir -p /app/data /app/volumes /app/backups

# Expose port
EXPOSE 8418

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DATA_PATH=/app/data
ENV VOLUMES_PATH=/app/volumes

# Set LD_LIBRARY_PATH to include the pip-installed NVIDIA libraries
# This is crucial for ctranslate2/faster-whisper to find cuDNN/cuBLAS
# Also includes nvidia-npp for torchcodec audio decoding
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/npp/lib:$LD_LIBRARY_PATH

# Run the application
CMD ["python", "-m", "app.main"]