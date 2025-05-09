# TTS Provider Server

A flexible WebSocket-based Text-to-Speech service that supports multiple TTS backends. Currently supports:

- Microsoft Edge TTS (default)
- Sesame CSM-1B
- Zonos TTS

## Installation

1. Clone this repository
2. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

### Installing Sesame CSM-1B Model

To use the Sesame CSM-1B model, you'll need to:

1. Login to Hugging Face (you need to accept the model terms):

   ```bash
   huggingface-cli login
   ```

2. Ensure access to models:
   The model will be automatically downloaded from Hugging Face when first used, but you need to have accepted the terms on the [Sesame CSM-1B model page](https://huggingface.co/sesame/csm-1b).

   Note: The model requires access to both `sesame/csm-1b` and `meta-llama/Llama-3.2-1B` on Hugging Face.

### Installing Zonos TTS Model

To use the Zonos TTS model, you'll need to:

1.  **Install System Dependencies:**
    Zonos requires `espeak-ng`. On Debian/Ubuntu, you can install it with:
    ```bash
    sudo apt-get update && sudo apt-get install -y espeak-ng
    ```
    (Note: The `Dockerfile` already includes this step.)

2.  **Python Dependencies:**
    The Zonos library will be installed automatically via `pip install -r requirements.txt`. It uses a specific fork (`UpperMoon0/nstut-zonos-fork`) which includes packaging fixes to ensure all submodules are correctly installed.

3.  **Reference Audio:**
    Zonos performs voice cloning using reference audio files. You need to place your `.wav` reference audio files in the `tts_models/zonos_reference_audio/` directory. For example:
    - `0.wav` or `default_speaker.wav` for speaker ID 0.
    - `1.wav` for speaker ID 1, etc.
    - `speaker_X.wav` can also be used for speaker ID X.
    Refer to `tts_models/zonos_tts.py` for more details on speaker mapping and file naming. An example reference audio file (`1.wav`) is provided.

## Running the Server

```bash
# Default command (uses Edge TTS by default if no model is specified in the request)
python -m run_server
```

Note: All TTS models (Edge, Sesame CSM-1B, Zonos) are loaded lazily. The server initializes, but the actual model weights are loaded into memory only when the first request requiring that specific model is received, or if preloading is triggered. This approach minimizes startup time and initial memory footprint.

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
- `zonos` - Use Zonos TTS model

## API Documentation

### WebSocket Request Format

Basic request format:

```json
{
  "text": "Text to convert to speech",
  "speaker": 0,
  "sample_rate": 24000,
  "model": "edge",  // Optional. Specifies model type (e.g., "edge", "sesame", "zonos"). Defaults to "edge" if not provided.
  "lang": "en-US"   // Optional. Specifies language. Defaults to "en-US".
}
```

#### Language Support (`lang` parameter)

You can specify the language for the TTS generation using the `lang` parameter.

- **Default**: `en-US` (if not specified)
- **Supported Languages**:
  - `en-US`: Supported by both `edge` and `sesame` models.
  - `ja-JP`: Supported only by the `edge` model.
    - Speaker 0: `ja-JP-KeitaNeural`
    - Speaker 1: `ja-JP-NanamiNeural`
    - *Note*: Speaker IDs 2 and 3 are not currently mapped for `ja-JP`. Requesting them will fall back to speaker 0 (`KeitaNeural`).

**Example (Japanese):**

```json
{
  "text": "こんにちは、これはテストです。",
  "speaker": 1,          // Use NanamiNeural voice
  "model": "edge",
  "lang": "ja-JP"
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

All TTS models are loaded lazily to optimize startup time and resource usage:

- The server initializes with a default model configuration (currently "edge").
- However, the actual loading of any model's weights and resources into memory occurs only when:
    1. The first WebSocket request that requires that specific model is received.
    2. An explicit preload operation is triggered (e.g., during server startup if configured, or via a specific command if implemented).
- If a request comes in for a model that isn't loaded yet, the request is queued, and the model loading process begins. Once loaded, queued requests for that model are processed.
- This ensures that only necessary models consume resources, and the server starts quickly.
