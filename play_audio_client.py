import asyncio
import websockets
import json
import argparse
import logging
import sounddevice as sd
import numpy as np
import io
import soundfile as sf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TTSStreamClient") # Renamed logger

async def stream_and_play_audio_from_server(host: str, port: int, text: str, model: str = None, speaker: int = 0):
    """
    Connects to the TTS server, requests speech generation in 'stream' mode,
    receives the audio stream, and plays it.
    """
    uri = f"ws://{host}:{port}"
    response_str_for_json_error = "" # For logging in case of JSONDecodeError

    try:
        async with websockets.connect(uri, max_size=None, ping_interval=None) as websocket: # Allow large messages
            logger.info(f"Connected to TTS server at {uri}")

            request_payload = {
                "text": text,
                "speaker": speaker
            }
            if model:
                request_payload["model"] = model
            
            logger.info(f"Sending TTS request: {json.dumps(request_payload)}")
            await websocket.send(json.dumps(request_payload))

            # 1. Receive initial JSON response from server (metadata or status)
            response_str_for_json_error = await websocket.recv()
            if not isinstance(response_str_for_json_error, str):
                logger.error(f"Expected initial JSON response string, but received binary data. Aborting.")
                logger.error(f"Received: {response_str_for_json_error[:100]}...")
                return
            
            logger.info(f"Received initial server response: {response_str_for_json_error}")
            response_data = json.loads(response_str_for_json_error)

            # Handle intermediate statuses like "loading", "queued"
            while response_data.get("status") in ["loading", "queued"]:
                logger.info(f"Server status: {response_data.get('status')}. Message: {response_data.get('message', 'Waiting...')}")
                # Wait for the next message, which should be the actual streaming start or an error
                response_str_for_json_error = await websocket.recv()
                if not isinstance(response_str_for_json_error, str):
                    logger.error(f"Expected JSON response string after '{response_data.get('status')}', but received binary. Aborting.")
                    return
                logger.info(f"Received updated server response: {response_str_for_json_error}")
                response_data = json.loads(response_str_for_json_error)

            # Now expect "streaming_start", "success" (with response_mode: "stream"), or an error status
            if response_data.get("status") == "streaming_start" or \
               (response_data.get("status") == "success" and response_data.get("response_mode") == "stream"):
                sample_rate = response_data.get("sample_rate")
                audio_format = response_data.get("format", "wav") # Default to wav, server should specify

                if not sample_rate:
                    logger.error("Server started streaming but did not provide 'sample_rate'. Cannot play audio.")
                    return

                logger.info(f"Streaming started. Sample rate: {sample_rate}, Format: {audio_format}")

                audio_buffer = io.BytesIO()
                try:
                    while True: # Loop to receive audio chunks
                        chunk = await websocket.recv() # This can be bytes or a JSON string for control messages

                        if isinstance(chunk, str):
                            logger.info(f"Received control message during stream: {chunk}")
                            try:
                                chunk_data = json.loads(chunk)
                                if chunk_data.get("status") == "streaming_end":
                                    logger.info("Streaming ended by server 'streaming_end' message.")
                                    break
                                elif chunk_data.get("status") == "error":
                                    logger.error(f"Server error during stream: {chunk_data.get('message', 'Unknown error')}")
                                    audio_buffer.close()
                                    return 
                                else:
                                    logger.warning(f"Received unknown JSON control message during stream: {chunk_data}")
                            except json.JSONDecodeError:
                                logger.warning(f"Received non-JSON string control message during audio stream: {chunk[:100]}...")
                        elif isinstance(chunk, bytes):
                            audio_buffer.write(chunk)
                        else:
                            logger.warning(f"Received unexpected message type during stream: {type(chunk)}. Breaking.")
                            break
                except websockets.exceptions.ConnectionClosed:
                    logger.info("Connection closed by server, assuming end of audio stream.")
                except Exception as e:
                    logger.error(f"Error receiving stream data: {e}")
                    audio_buffer.close()
                    return

                audio_buffer.seek(0)
                if audio_buffer.getbuffer().nbytes > 0:
                    logger.info(f"Audio stream received: {audio_buffer.getbuffer().nbytes} bytes.")
                    try:
                        data, samplerate_from_file = sf.read(audio_buffer, dtype='float32')
                        
                        if sample_rate != samplerate_from_file:
                             logger.warning(f"Sample rate from metadata ({sample_rate}) differs from file/stream ({samplerate_from_file}). Using {samplerate_from_file}.")
                        
                        logger.info(f"Playing audio with sample rate: {samplerate_from_file}")
                        sd.play(data, samplerate_from_file)
                        sd.wait() # Wait until playback is finished
                        logger.info("Audio playback finished.")
                    except Exception as e:
                        logger.error(f"Error playing audio: {e}")
                        logger.error("Please ensure 'sounddevice', 'numpy', and 'soundfile' are installed.")
                        logger.error("You can typically install them using: pip install sounddevice numpy soundfile")
                        logger.error("Also, make sure you have a working sound output device.")
                else:
                    logger.warning("No audio data received in buffer to play.")
                audio_buffer.close()

            elif response_data.get("status") == "error":
                error_msg = response_data.get("message", "Unknown error from server")
                logger.error(f"Server error: {error_msg}")
            else: # Initial response (or response after loading/queued) wasn't handled
                logger.error(f"Unexpected server response status: {response_data.get('status')}. Full response: {response_data}")

    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"Connection closed unexpectedly: {e}")
    except ConnectionRefusedError:
        logger.error(f"Connection refused. Is the server running at ws://{host}:{port}?")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response: {e}. Received: {response_str_for_json_error[:200] if isinstance(response_str_for_json_error, str) else 'Non-string data'}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS Client to request speech generation in 'stream' mode, receive, and play the audio.")
    parser.add_argument("text", type=str, help="Text to synthesize.")
    parser.add_argument("--model", type=str, default=None, help="TTS model to use (e.g., 'edge', 'sesame'). Server default if not specified.")
    parser.add_argument("--speaker", type=int, default=0, help="Speaker ID to use (e.g., 0 for default male, 1 for default female). Refer to server's SPEAKER_MAPPING.")
    parser.add_argument("--host", type=str, default="localhost", help="Server host.")
    parser.add_argument("--port", type=int, default=9000, help="Server port.")
    
    args = parser.parse_args()

    logger.info("Attempting to stream and play audio. Ensure 'sounddevice', 'numpy', and 'soundfile' are installed (e.g., pip install sounddevice numpy soundfile).")
    asyncio.run(stream_and_play_audio_from_server(args.host, args.port, args.text, args.model, args.speaker))
