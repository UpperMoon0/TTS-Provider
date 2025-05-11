# Stage 1: Builder
# Use the -devel image which includes CUDA compiler (nvcc) for dependencies like Triton
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=True
ENV DEBIAN_FRONTEND=noninteractive
ENV VENV_PATH=/opt/venv

# Install Python 3.12, pip, git, and other system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    # python3.12-venv is removed as it's not used in the builder stage
    git \
    ffmpeg \
    espeak-ng && \
    # Install pip for Python 3.12
    python3.12 -m ensurepip --upgrade && \
    # Remove software-properties-common after adding PPA
    apt-get purge -y --auto-remove software-properties-common && \
    # Clean up apt cache
    rm -rf /var/lib/apt/lists/*

# Make python3.12 the default python3 and pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    python3 -m pip install --no-cache-dir --upgrade pip

# Upgrade pip, setuptools, wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
# Using --no-cache-dir here and for other pip installs to reduce layer size
RUN pip install --no-cache-dir --resume-retries 5 torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121

# Copy requirements.txt and install remaining dependencies
# Triton (if not commented out in requirements.txt) would be compiled here using nvcc from the devel image.
COPY requirements.txt .
# Create a temporary requirements file without torch/torchaudio (already installed)
# and without development dependencies (pytest, etc.) to reduce image size.
RUN grep -vE '^torch==|^torchaudio==|^pytest==|^pytest-asyncio==|^pytest-cov==|^pytest-mock' requirements.txt > /requirements_no_torch_dev.txt
RUN pip install --no-cache-dir --ignore-installed blinker -r /requirements_no_torch_dev.txt
RUN rm /requirements_no_torch_dev.txt

# Stage 2: Final Runtime Image
# Use the -runtime image which is smaller
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=True
ENV TTS_HOST=0.0.0.0
ENV TTS_PORT=9000
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV DEBIAN_FRONTEND=noninteractive
# Define Hugging Face cache directory
ENV HF_HOME=/app/huggingface_cache

# Install Python 3.12 runtime and essential system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    ffmpeg \
    espeak-ng && \
    apt-get purge -y --auto-remove software-properties-common && \
    rm -rf /var/lib/apt/lists/*

# Make python3.12 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Copy installed Python packages and executables from the builder stage
COPY --from=builder /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin
# Ensure the main python3.12 interpreter and its symlink are correctly in place if not already handled by apt install
COPY --from=builder /usr/bin/python3.12 /usr/bin/python3.12
COPY --from=builder /usr/bin/python3 /usr/bin/python3

WORKDIR /app

# Copy the rest of the application code into the container
# This respects the .dockerignore file
COPY . .

# Copy the entrypoint script and make it executable
COPY entrypoint.sh .
RUN chmod +x /app/entrypoint.sh

# Make port available
EXPOSE ${TTS_PORT}

# Set the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD []
