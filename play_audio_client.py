import asyncio
import websockets
import json
import argparse
import io
import logging

# Attempt to import sounddevice and soundfile, provide instructions if missing
try:
    import sounddevice as sd
    import soundfile as sf
    SOUND_LIBS_AVAILABLE = True
except ImportError:
    SOUND_LIBS_AVAILABLE = False
    print("Warning: 'sounddevice' or 'soundfile' library not found. Playback will be skipped.")
    print("Please install them to enable audio playback: pip install sounddevice soundfile numpy")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PlayAudioClient")

async def play_audio_from_server(host: str, port: int, text: str, model: str = None, speaker: int = 0):
    """
    Connects to the TTS server, requests speech generation,
    receives the audio, and plays it.
    """
    uri = f"ws://{host}:{port}"
    
    try:
        async with websockets.connect(uri, max_size=None, ping_interval=None) as websocket: # Allow large messages
            logger.info(f"Connected to TTS server at {uri}")

            request_payload = {
                "text": text,
                "speaker": speaker,
                "response_mode": "stream" # We want binary data directly
            }
            if model:
                request_payload["model"] = model
            
            logger.info(f"Sending TTS request: {json.dumps(request_payload)}")
            await websocket.send(json.dumps(request_payload))

            # 1. Receive JSON metadata
            metadata_str = await websocket.recv()
            if not isinstance(metadata_str, str):
                logger.error(f"Expected JSON metadata string, but received binary data first. Aborting.")
                logger.error(f"Received: {metadata_str[:100]}...")
                return

            logger.info(f"Received metadata: {metadata_str}")
            metadata = json.loads(metadata_str)

            if metadata.get("status") != "success":
                error_msg = metadata.get("message", "Unknown error from server")
                logger.error(f"Server error: {error_msg}")
                if metadata.get("status") == "loading" or metadata.get("status") == "queued":
                    logger.info("Model is loading or request is queued. Waiting for processing...")
                    # The server sends another metadata message when processing starts/finishes
                    metadata_str_updated = await websocket.recv()
                    if not isinstance(metadata_str_updated, str):
                        logger.error("Expected updated JSON metadata, but received binary. Aborting.")
                        return
                    logger.info(f"Received updated metadata: {metadata_str_updated}")
                    metadata = json.loads(metadata_str_updated)
                    if metadata.get("status") != "success":
                        error_msg = metadata.get("message", "Unknown error after queue/load")
                        logger.error(f"Server error after queue/load: {error_msg}")
                        return
                else:
                    return # Original error was not loading/queued

            # 2. Receive audio data
            audio_bytes_list = []
            expected_length = metadata.get("length_bytes")
            if expected_length is None:
                logger.error("Metadata did not contain 'length_bytes'. Cannot reliably receive audio.")
                return

            received_length = 0
            logger.info(f"Expecting {expected_length} bytes of audio data.")

            while received_length < expected_length:
                message = await websocket.recv()
                if isinstance(message, bytes):
                    audio_bytes_list.append(message)
                    received_length += len(message)
                    logger.info(f"Received {len(message)} audio bytes. Total: {received_length}/{expected_length}")
                elif isinstance(message, str):
                    # This might be an error message or unexpected JSON
                    logger.warning(f"Received unexpected string message while expecting audio: {message}")
                    # You might want to break or handle this case differently
                if received_length >= expected_length:
                    break
            
            if not audio_bytes_list:
                logger.error("No audio data received.")
                return

            full_audio_data = b"".join(audio_bytes_list)

            if len(full_audio_data) != expected_length:
                logger.warning(f"Received {len(full_audio_data)} bytes, but expected {expected_length} bytes. Audio might be incomplete or corrupted.")


            logger.info(f"Successfully received {len(full_audio_data)} bytes of audio data.")

            if SOUND_LIBS_AVAILABLE:
                try:
                    logger.info("Playing audio...")
                    # Use io.BytesIO to treat the byte string as a file
                    audio_file = io.BytesIO(full_audio_data)
                    data, samplerate = sf.read(audio_file)
                    sd.play(data, samplerate)
                    sd.wait() # Wait until playback is finished
                    logger.info("Playback finished.")
                except Exception as e:
                    logger.error(f"Error playing audio: {e}")
            else:
                logger.info("Audio received but playback libraries are not available. Skipping playback.")
                # Optionally, save to a file here if playback is not available
                output_filename = "received_audio.wav"
                with open(output_filename, "wb") as f:
                    f.write(full_audio_data)
                logger.info(f"Audio saved to {output_filename} as playback is unavailable.")

    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"Connection closed unexpectedly: {e}")
    except ConnectionRefusedError:
        logger.error(f"Connection refused. Is the server running at ws://{host}:{port}?")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS Client to request speech and play audio.")
    parser.add_argument("text", type=str, help="Text to synthesize.")
    parser.add_argument("--model", type=str, default=None, help="TTS model to use (e.g., 'edge', 'sesame'). Server default if not specified.")
    parser.add_argument("--speaker", type=int, default=0, help="Speaker ID to use (e.g., 0 for default male, 1 for default female). Refer to server's SPEAKER_MAPPING.")
    parser.add_argument("--host", type=str, default="localhost", help="Server host.")
    parser.add_argument("--port", type=int, default=9000, help="Server port.")
    
    args = parser.parse_args()

    asyncio.run(play_audio_from_server(args.host, args.port, args.text, args.model, args.speaker))
