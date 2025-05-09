FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=True
# Set default host and port, can be overridden at runtime
ENV TTS_HOST=0.0.0.0
ENV TTS_PORT=9000

WORKDIR /app

# Install system dependencies (git is needed for silentcipher, ffmpeg for pydub)
# Clean up apt cache afterwards to keep image size down
RUN apt-get update && \
    apt-get install -y --no-install-recommends git ffmpeg espeak-ng && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Use --no-cache-dir to reduce image size
# Note: Triton installation might be complex and require specific dependencies
# or a different base image if GPU support is needed. This assumes CPU usage.
RUN pip install --no-cache-dir -r requirements.txt

# This includes the Sesame-CSM-1b-Folk directory and all .py files
COPY . .

COPY entrypoint.sh .
RUN chmod +x /app/entrypoint.sh

EXPOSE ${TTS_PORT}

ENTRYPOINT ["/app/entrypoint.sh"]

# Default command (can be overridden). Arguments passed here will be appended to the entrypoint command via "$@"
# Since run_server.py handles defaults via argparse/env vars, an empty CMD is suitable.
CMD []
