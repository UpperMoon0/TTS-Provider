# Use a single stage build to avoid slow COPY operations of large PyTorch files
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=True
ENV TTS_HOST=0.0.0.0
ENV TTS_PORT=9000
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/app/huggingface_cache
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Install Python, build dependencies, and runtime dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    gnupg \
    wget \
    git \
    build-essential \
    ffmpeg \
    espeak-ng && \
    # Add deadsnakes PPA for Python 3.12
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    # Install Python 3.12 and development headers
    apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv && \
    # Install pip
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.12 get-pip.py && \
    rm get-pip.py && \
    # Set Python 3.12 as default
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    python3 -m pip install --upgrade pip setuptools wheel && \
    # Cleanup initial setup tools
    apt-get purge -y --auto-remove software-properties-common gnupg && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch (Cached)
# Installing this before other requirements allows caching this heavy layer
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --resume-retries 5 torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install production requirements
COPY requirements-prod.txt .
# We remove nvidia-*, torch, and torchaudio packages because they are already installed via the cached PyTorch layer
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -vE "^nvidia-|^torch==|^torchaudio==" requirements-prod.txt > requirements-prod-filtered.txt && \
    pip install --ignore-installed blinker -r requirements-prod-filtered.txt

# Cleanup build dependencies to reduce image size
# Note: We keep runtime dependencies like ffmpeg, espeak-ng, python3.12
RUN apt-get purge -y git build-essential python3.12-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy application code
COPY . .

# Setup entrypoint
COPY entrypoint.sh .
RUN chmod +x /app/entrypoint.sh

# Expose port
EXPOSE ${TTS_PORT}

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
CMD []
