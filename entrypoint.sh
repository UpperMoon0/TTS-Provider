#!/bin/sh
# Exit immediately if a command exits with a non-zero status.
set -e

# Python 3.12 should be the default python3 on the system PATH
PYTHON_EXEC="python3"

# Print the command we are about to execute (optional, for debugging)
echo "Running command: ${PYTHON_EXEC} run_server.py --host ${TTS_HOST} --port ${TTS_PORT} $@"

# Execute the python script, replacing the shell process with the python process.
# This ensures that signals like SIGTERM are passed directly to the python script.
exec ${PYTHON_EXEC} run_server.py --host "${TTS_HOST}" --port "${TTS_PORT}" "$@"
