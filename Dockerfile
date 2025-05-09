# 1. Base Image: NVIDIA CUDA with a compatible CUDA version for PyTorch 2.6
# Using CUDA 12.1 as PyTorch 2.6.0 has pre-built wheels for cu121
# Using a -devel image includes the CUDA compiler (nvcc) and full libraries,
# which can be helpful for dependencies like Triton that might compile extensions.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=True
# Set default host and port, can be overridden at runtime
ENV TTS_HOST=0.0.0.0
ENV TTS_PORT=9000
# Environment variables for NVIDIA Container Toolkit
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# 2. Install Python 3.12, pip, and other system dependencies
# Ubuntu 22.04's default python3 is older.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    python3-pip \
    python3.12-venv \
    git \
    ffmpeg \
    espeak-ng && \
    # Clean up apt cache
    rm -rf /var/lib/apt/lists/*

# Make python3.12 the default python3 and ensure pip points to python3.12's pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    python3 -m pip install --no-cache-dir --upgrade pip

WORKDIR /app

# 3. Install PyTorch with CUDA support
# Install torch 2.6.0 and torchaudio 2.6.0 for CUDA 12.1
RUN python3 -m pip install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121

# 4. Copy requirements.txt and install other dependencies
# We'll filter out torch and torchaudio as they are already installed with CUDA support.
COPY requirements.txt .
# Create a temporary requirements file without torch/torchaudio
RUN grep -vE '^torch==|^torchaudio==' requirements.txt > /app/requirements_no_torch.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements_no_torch.txt
RUN rm /app/requirements_no_torch.txt # Clean up the temporary file

# Note: The 'triton' package from requirements.txt should now install the Linux/GPU version.
# The CUDA toolkit from the base image should provide necessary components for it.

# 5. Copy the rest of the application code into the container
# This respects the .dockerignore file
COPY . .

# 6. Copy the entrypoint script and make it executable
COPY entrypoint.sh .
RUN chmod +x /app/entrypoint.sh

# Make port available
EXPOSE ${TTS_PORT}

# Set the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD []
