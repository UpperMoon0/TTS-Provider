# TTS Provider Server

A flexible WebSocket-based Text-to-Speech service that supports multiple TTS backends. Currently supports:

- Microsoft Edge TTS (default)
- Sesame CSM-1B

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Server

### Default (Microsoft Edge TTS)

Run the server with Microsoft Edge TTS as the default model (this is the default configuration):

```bash
# Default command (uses Edge TTS)
python -m run_server

# Or explicitly specify Edge TTS
python -m run_server --model edge
```

### Using Sesame CSM-1B

Run the server with Sesame CSM-1B as the TTS model:

```bash
# Specify Sesame as the model
python -m run_server --model sesame
```

Note: The Sesame CSM-1B model is loaded on-demand when the first request that uses it is received, not at startup. This helps reduce startup time and memory usage until the model is actually needed.

## Client Usage

Clients can connect to the server via WebSocket. See `tts_client.py` for a complete client implementation.

### Basic Example

```python
import asyncio
from tts_client import TTSClient

async def main():
    client = TTSClient(host="localhost", port=9000)
    
    try:
        await client.connect()
        
        # Get server information
        info = await client.get_server_info()
        print(f"Available models: {info.get('available_models')}")
        
        # Generate speech with default model (Edge TTS)
        await client.generate_speech(
            text="Hello, this is a test.",
            output_path="output.wav"
        )
        
        # Generate speech with specific model
        await client.generate_speech(
            text="Hello, this is Sesame CSM speaking.",
            output_path="output_sesame.wav",
            model="sesame"
        )
        
        # Generate speech with Edge TTS
        await client.generate_speech(
            text="Hello, this is Edge TTS speaking.",
            output_path="output_edge.wav",
            model="edge",
            speaker=2  # Use the Davis voice
        )
        
    finally:
        await client.disconnect()

asyncio.run(main())
```

## Speaker ID Mapping

The TTS Provider supports a unified speaker ID system across different models. You can use the same integer speaker IDs (0-3) regardless of which model you're using.

- **Simple usage**: Just provide a speaker ID as an integer, and it will be automatically mapped to the appropriate voice based on the model being used.
- **Cross-model consistency**: The same speaker IDs work with both Sesame CSM and Edge TTS models.

### Speaker ID Reference Table

| ID | Description | Sesame CSM | Edge TTS |
|----|-------------|------------|----------|
| 0 | Default Male Voice | Male Voice | US Male (Guy) |
| 1 | Default Female Voice | Female Voice | US Female (Jenny) |
| 2 | Alternative Male Voice | Male Voice | US Male (Davis) |
| 3 | Alternative Female Voice | Female Voice | UK Female (Sonia) |

*Note*: For Sesame CSM, male voices (0 and 2) both map to speaker 0, and female voices (1 and 3) both map to speaker 1, since Sesame only supports two distinct voices.

## Selecting Models

Clients can select which model to use in each request by including a `model` parameter:

- `sesame` (or `csm`) - Use Sesame CSM-1B model
- `edge` (or `edge-tts`) - Use Microsoft Edge TTS

## API Documentation

### WebSocket Request Format

Basic request format:

```json
{
  "text": "Text to convert to speech",
  "speaker": 0,  
  "sample_rate": 24000,
  "response_mode": "stream",
  "model": "edge"  // Optional, specify model type (edge or sesame)
}
```

#### Edge TTS Voice Selection

When using Edge TTS, you can only specify which voice to use via the speaker ID. The Edge TTS implementation uses only default voice parameters - no customization of rate, volume or pitch is allowed:

```json
{
  "text": "Text to convert to speech",
  "speaker": 0,
  "model": "edge"
}
```

**Important Note:** For Edge TTS, voice modification parameters like `rate`, `volume`, and `pitch` are not supported and will be ignored. Edge TTS will always use the natural, default voice characteristics to ensure maximum reliability and consistent sound quality.

### Server Information Request

To get information about the server and available models:

```json
{
  "command": "info"
}
```

The response includes the available speaker mappings to help you select the appropriate voice.

## Model Loading Behavior

- **Edge TTS**: Loaded immediately at startup since it's lightweight
- **Sesame CSM-1B**: Loaded on-demand when the first request using it is received
