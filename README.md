# TTS Provider Server

A flexible WebSocket-based Text-to-Speech service that supports multiple TTS backends. Currently supports:

- Sesame CSM-1B (default)
- Microsoft Edge TTS

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Server

### Default (Sesame CSM-1B)

Run the server with Sesame CSM-1B as the default model:

```bash
# On Linux/Mac
./run_with_sesame.sh

# Or directly
python -m run_server
```

### Using Edge TTS

Run the server with Microsoft Edge TTS as the default model:

```bash
# On Linux/Mac
./run_with_edge_tts.sh

# Or directly
python -m run_server --model edge
```

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
        
        # Generate speech with default model
        await client.generate_speech(
            text="Hello, this is a test.",
            output_path="output.wav"
        )
        
        # Generate speech with specific model
        await client.generate_speech(
            text="Hello, this is Edge TTS speaking.",
            output_path="output_edge.wav",
            model="edge",
            # Edge TTS specific parameters
            rate="+10%",  # 10% faster
            volume="+20%"  # 20% louder
        )
        
    finally:
        await client.disconnect()

asyncio.run(main())
```

## Speaker ID Mapping

The TTS Provider supports a unified speaker ID system across different models. You can use the same integer speaker IDs (0-3) regardless of which model you're using.

- **Simple usage**: Just provide a speaker ID as an integer (0-3), and it will be automatically mapped to the appropriate voice based on the model being used.
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
  "model_type": "sesame"  // Optional, specify model type
}
```

#### Edge TTS Specific Parameters

When using Edge TTS, you can include additional parameters:

```json
{
  "text": "Text to convert to speech",
  "speaker": 0,
  "model_type": "edge",
  "rate": "+10%",
  "volume": "+20%",
  "pitch": "-5%"
}
```

### Server Information Request

To get information about the server and available models:

```json
{
  "command": "info"
}
```

The response includes the available speaker mappings to help you select the appropriate voice.
