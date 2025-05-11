# TTS Provider Server

A flexible WebSocket-based Text-to-Speech service that supports multiple TTS backends. Currently supports:

- Microsoft Edge TTS (default)
- Sesame CSM-1B
- Zonos TTS

## Installation

1. Clone this repository
2. Install the required packages:

   ```bash
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

1. **Install System Dependencies:**
    Zonos requires `espeak-ng`. On Debian/Ubuntu, you can install it with:

    ```bash
    sudo apt-get update && sudo apt-get install -y espeak-ng
    ```

    (Note: The `Dockerfile` already includes this step.)

2. **Python Dependencies:**
    The Zonos library will be installed automatically via `pip install -r requirements.txt`. It uses a specific fork (`UpperMoon0/nstut-zonos-fork`) which includes packaging fixes to ensure all submodules are correctly installed.

3. **Reference Audio:**
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

## Running with Docker

You can also run the TTS Provider server using Docker.

1. **Pull the Docker Image:**

    ```bash
    docker pull nstut/tts-provider
    ```

2. **Run the Docker Container:**

    ```bash
    docker run --rm -itd --gpus all --name TTS-Provider -p 9000:9000 -e HF_TOKEN=<YOUR_HF_TOKEN> nstut/tts-provider
    ```

    **Explanation of the command:**
    - `--rm`: Automatically remove the container when it exits.
    - `-itd`: Run in interactive, TTY, and detached (background) mode.
    - `--gpus all`: (Optional) If you have NVIDIA GPUs and want to use them for models like Sesame CSM and Zonos, this flag enables GPU access. Remove if you don't have GPUs or don't need GPU support.
    - `--name TTS-Provider`: Assigns a name to the container for easier management.
    - `-p 9000:9000`: Maps port 9000 on your host to port 9000 in the container.
    - `-e HF_TOKEN=<YOUR_HF_TOKEN>`: Sets the Hugging Face token as an environment variable. **Replace `<YOUR_HF_TOKEN>` with your actual Hugging Face token.** This is required if you plan to use models like Sesame CSM-1B or Zonos that need to be downloaded from Hugging Face.
    - `nstut/tts-provider`: The name of the Docker image to run.

    The server will then be accessible at `ws://localhost:9000`.

### Persisting Downloaded Models with Docker Volumes

By default, when a Docker container is removed, any data written inside it (like downloaded Hugging Face models) is lost. To prevent re-downloading models every time you start a new container, you should use a Docker volume to persist the Hugging Face cache.

The `Dockerfile` is configured to use `/app/huggingface_cache` as the Hugging Face home directory (`HF_HOME`). You can mount a volume to this location:

**1. Using a Named Volume (Recommended):**

First, create a named volume if you haven't already:

```bash
docker volume create tts_provider_hf_cache
```

Then, run your container, mounting this volume:

```bash
docker run --rm -itd --gpus all --name TTS-Provider \
  -p 9000:9000 \
  -e HF_TOKEN=<YOUR_HF_TOKEN> \
  -v tts_provider_hf_cache:/app/huggingface_cache \
  nstut/tts-provider
```

This will store the downloaded models in the `tts_provider_hf_cache` volume, and they will be available to subsequent containers that mount the same volume.

**2. Using a Host Directory (Bind Mount):**

Alternatively, you can map a directory from your host machine into the container:

```bash
docker run --rm -itd --gpus all --name TTS-Provider \
  -p 9000:9000 \
  -e HF_TOKEN=<YOUR_HF_TOKEN> \
  -v /path/on/your/host/hf_cache:/app/huggingface_cache \
  nstut/tts-provider
```

Replace `/path/on/your/host/hf_cache` with an actual directory path on your computer.

Using either of these methods will ensure that models downloaded by Hugging Face (for both Sesame CSM and Zonos) are cached persistently.

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

| ID | General Description | Sesame CSM | Edge TTS | Zonos TTS |
|----|---------------------|------------|----------|-----------|
| 0  | Primary/Default     | Male Voice | US Male (Guy) | Cloned (e.g., `0.wav`/`default_speaker.wav`) |
| 1  | Secondary           | Female Voice | US Female (Jenny) | Cloned (e.g., `1.wav`) |
| 2  | Tertiary            | Male Voice | US Male (Davis) | Cloned (e.g., `2.wav`) |
| 3  | Quaternary          | Female Voice | UK Female (Sonia) | Cloned (e.g., `3.wav`) |
| ...| Additional Voices   | N/A        | N/A      | Cloned (e.g., `X.wav`/`speaker_X.wav`) |

*Note for Sesame CSM*: Speaker IDs 0 and 2 map to its male voice; IDs 1 and 3 map to its female voice.
*Note for Zonos TTS*: Speaker IDs correspond to user-provided `.wav` files in the `tts_models/zonos_reference_audio/` directory (e.g., speaker ID `X` typically maps to `X.wav` or `speaker_X.wav`). The voice characteristics are determined by these reference files. The system can support many such cloned speakers.

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

Clients **should** specify the language for TTS generation using standard IETF language tags (e.g., `en-US`, `ja-JP`, `es-ES`) in the `lang` parameter of the WebSocket request.

- **Default**: If the `lang` parameter is not provided, `en-US` is generally assumed by most models, though specific model behavior can vary.
- **Server-Side Language Code Mapping**:
  - The server-side TTS models (`edge`, `sesame`, `zonos`) are responsible for mapping these standard input language codes to the specific formats required by their underlying TTS engines. This mapping is handled by a `_map_language_code` method within each model.
  - While models *may* attempt to normalize and map common variations (e.g., "en", "english" to "en-US"), relying on this is discouraged for client implementations.
- **Error Handling**:
  - If a model cannot map the provided `lang` parameter to a supported language code (even after its internal normalization attempts), the server will return an error, and speech generation will fail. This indicates the language is not supported by the chosen model.
- **Model-Specific Support**:
  - **EdgeTTS**: Accepts standard codes like "en-US", "ja-JP". See its `VOICE_MAPPINGS` for explicitly configured target languages.
  - **SesameCSM**: Primarily supports "en-US". Requests for other language codes will result in an error.
  - **ZonosTTS**: Accepts standard codes and maps them to its wide range of supported languages (e.g., "en-US" might map to "en-us", "ja-JP" to "ja"). It uses a comprehensive mapping (see `PREFERRED_ZONOS_LANG_MAP` in `zonos_tts.py`) and checks against dynamically available Zonos language codes.
- **Client Recommendation**: For maximum compatibility and predictability, clients **must** send well-formed IETF language tags (e.g., `en-US`, `ja-JP`).

**Example (Japanese with EdgeTTS):**

```json
{
  "text": "こんにちは、これはテストです。",
  "speaker": 1,
  "model": "edge",
  "lang": "ja-JP" // Client sends standard ja-JP
}
```

**Example (English with Zonos):**

```json
{
  "text": "Hello, this is Zonos.",
  "speaker": 0, 
  "model": "zonos",
  "lang": "en-US" // Client sends standard en-US; Zonos maps to "en-us" or similar
}
```

**Important Note:** For Edge TTS, voice modification parameters like `rate`, `volume`, and `pitch` are not supported and will be ignored. Edge TTS will always use the default voice characteristics.

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
