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

## Running the Server

Start the WebSocket server:
```bash
python run_server.py [--host HOST] [--port PORT]
```

Arguments:
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to listen on (default: 8765)

Example:
```bash
# Default settings
python run_server.py

# Custom port
python run_server.py --port 8080
```

The server will automatically download and load the CSM-1B model on first run. The model will be stored locally in the `models` directory within the project.

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
