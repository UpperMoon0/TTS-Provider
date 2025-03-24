# TTS Provider Server

A WebSocket server that provides Text-to-Speech services using CSM-1B model.

## Setup

1. Clone the CSM repository inside this project:
```bash
git clone https://github.com/SesameAILabs/csm.git
```

2. Create and activate a Python virtual environment (Python 3.10 required):
```bash
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

4. Log in to Hugging Face (required to download the model):
```bash
huggingface-cli login
```

## Model Management

The server supports two modes for loading the model:

1. Download Mode (default):
   - Downloads the model from HuggingFace when first run
   - Requires HF_TOKEN to be set in .env file
   - Model is stored in HF_HOME/hub directory

2. Reuse Mode:
   - Uses an existing downloaded model
   - No need to download again if you already have the model files
   - Must specify the path to the model folder

Create a `.env` file (copy from `.env.template`) to configure model storage:
```
HF_TOKEN=your_huggingface_token_here
HF_HOME=D:/Dev/Models/huggingface
```

## Running the Server

Start the WebSocket server:
```bash
python run_server.py [--host HOST] [--port PORT] [--mode MODE] [--model-path PATH]
```

Arguments:
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to listen on (default: 8765)
- `--mode`: Model loading mode: "download" or "reuse" (default: download)
- `--model-path`: Path to existing model folder when using reuse mode
                Example: "D:/Dev/Models/huggingface/hub/models--sesame--csm-1b"

Examples:
```bash
# Download mode (default)
python run_server.py

# Reuse existing model
python run_server.py --mode reuse --model-path "D:/Dev/Models/huggingface/hub/models--sesame--csm-1b"

# Custom port
python run_server.py --port 8080
```

The server will load the CSM-1B model at startup before accepting any WebSocket connections.

## WebSocket API

Send requests to the server in the following JSON format:
```json
{
    "text": "Text to convert to speech",
    "speaker": 0  # 0 for male voice, 1 for female voice (optional, defaults to 0)
}
```

The server will respond with:
1. A JSON message containing metadata:
```json
{
    "status": "success",
    "message": "Audio generated successfully",
    "format": "wav",
    "sample_rate": 24000,
    "length_bytes": <size of the audio data>
}
```

2. The WAV audio data as a binary message

### Loading Status

If the model is still loading when you connect, you'll receive:
```json
{
    "status": "loading",
    "message": "TTS model is still loading, request will be processed when ready"
}
```

The server will process your request automatically once the model is loaded.
