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

## Speaker IDs

### Sesame CSM-1B Speakers
- 0: Male voice
- 1: Female voice

### Edge TTS Speakers
- 0: US Male (Guy)
- 1: US Female (Jenny)
- 2: US Female (Aria)
- 3: UK Male (Ryan)
- 4: UK Female (Sonia)
- 5-10: Various international voices
