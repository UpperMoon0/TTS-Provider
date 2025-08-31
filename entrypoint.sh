#!/bin/sh
# Exit immediately if a command exits with a non-zero status.
set -e

# Python 3.12 should be the default python3 on the system PATH
PYTHON_EXEC="python3"

# Print the command we are about to execute (optional, for debugging)
echo "Running command: ${PYTHON_EXEC} main.py --host ${TTS_HOST} --port ${TTS_PORT} $@"

# Check CUDA availability
echo "Checking CUDA availability with PyTorch..."
${PYTHON_EXEC} -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Device Count: {torch.cuda.device_count()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'PyTorch Version: {torch.__version__}')"
echo "CUDA check complete."

# Execute the python script, replacing the shell process with the python process.
# This ensures that signals like SIGTERM are passed directly to the python script.
exec ${PYTHON_EXEC} main.py --host "${TTS_HOST}" --port "${TTS_PORT}" "$@"