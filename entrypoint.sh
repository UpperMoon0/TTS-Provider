#!/bin/sh
# Exit immediately if a command exits with a non-zero status.
set -e

# VENV_PATH should be set by the Dockerfile. Default if not set.
VENV_PATH="${VENV_PATH:-/opt/venv}"

# Activate the virtual environment
if [ -f "${VENV_PATH}/bin/activate" ]; then
  . "${VENV_PATH}/bin/activate"
  echo "Virtual environment activated from ${VENV_PATH}"
else
  echo "Error: Virtual environment activate script not found at ${VENV_PATH}/bin/activate"
  # Attempt to run without explicit activation, relying on PATH from Dockerfile
  # This provides a fallback but logs a warning.
  echo "Warning: Proceeding without explicit venv activation. Ensure PATH is correctly set in Dockerfile."
fi

# Print the command we are about to execute (optional, for debugging)
echo "Running command: python run_server.py --host ${TTS_HOST} --port ${TTS_PORT} $@"

# Execute the python script, replacing the shell process with the python process.
# This ensures that signals like SIGTERM are passed directly to the python script.
# python should resolve to the venv's python due to PATH or activation.
exec python run_server.py --host "${TTS_HOST}" --port "${TTS_PORT}" "$@"
