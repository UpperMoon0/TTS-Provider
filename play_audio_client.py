import asyncio
import websockets
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TTSFileClient")

async def get_audio_file_from_server(host: str, port: int, text: str, model: str = None, speaker: int = 0):
    """
    Connects to the TTS server, requests speech generation in 'file' mode,
    and receives the path to the generated audio file on the server.
    """
    uri = f"ws://{host}:{port}"
    
    try:
        async with websockets.connect(uri, max_size=None, ping_interval=None) as websocket: # Allow large messages
            logger.info(f"Connected to TTS server at {uri}")

            request_payload = {
                "text": text,
                "speaker": speaker,
                "response_mode": "file" # Request server to save file and return path
            }
            if model:
                request_payload["model"] = model
            
            logger.info(f"Sending TTS request: {json.dumps(request_payload)}")
            await websocket.send(json.dumps(request_payload))

            # 1. Receive JSON response from server (which includes file path)
            response_str = await websocket.recv()
            if not isinstance(response_str, str):
                logger.error(f"Expected JSON response string, but received binary data. Aborting.")
                logger.error(f"Received: {response_str[:100]}...")
                return

            logger.info(f"Received server response: {response_str}")
            response_data = json.loads(response_str)

            if response_data.get("status") == "success" and response_data.get("response_mode") == "file":
                filename = response_data.get("filename")
                filepath_on_server = response_data.get("filepath")
                if filename and filepath_on_server:
                    logger.info(f"Server successfully generated audio and saved it to: {filepath_on_server} (filename: {filename})")
                else:
                    logger.error("Server response success, but filename or filepath missing in 'file' mode.")
            elif response_data.get("status") == "loading" or response_data.get("status") == "queued":
                logger.info("Model is loading or request is queued. Waiting for processing...")
                # The server sends another metadata message when processing starts/finishes or if an error occurs
                updated_response_str = await websocket.recv()
                if not isinstance(updated_response_str, str):
                    logger.error("Expected updated JSON response, but received binary. Aborting.")
                    return
                logger.info(f"Received updated server response: {updated_response_str}")
                updated_response_data = json.loads(updated_response_str)
                if updated_response_data.get("status") == "success" and updated_response_data.get("response_mode") == "file":
                    filename = updated_response_data.get("filename")
                    filepath_on_server = updated_response_data.get("filepath")
                    if filename and filepath_on_server:
                        logger.info(f"Server successfully generated audio (after queue/load) and saved it to: {filepath_on_server} (filename: {filename})")
                    else:
                        logger.error("Server response success (after queue/load), but filename or filepath missing.")
                else:
                    error_msg = updated_response_data.get("message", "Unknown error after queue/load")
                    logger.error(f"Server error after queue/load: {error_msg}")
            else:
                error_msg = response_data.get("message", "Unknown error from server")
                logger.error(f"Server error: {error_msg}")

    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"Connection closed unexpectedly: {e}")
    except ConnectionRefusedError:
        logger.error(f"Connection refused. Is the server running at ws://{host}:{port}?")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS Client to request speech generation in 'file' mode and get the server-side file path.")
    parser.add_argument("text", type=str, help="Text to synthesize.")
    # --output argument is removed as the server handles file saving and naming.
    parser.add_argument("--model", type=str, default=None, help="TTS model to use (e.g., 'edge', 'sesame'). Server default if not specified.")
    parser.add_argument("--speaker", type=int, default=0, help="Speaker ID to use (e.g., 0 for default male, 1 for default female). Refer to server's SPEAKER_MAPPING.")
    parser.add_argument("--host", type=str, default="localhost", help="Server host.")
    parser.add_argument("--port", type=int, default=9000, help="Server port.")
    
    args = parser.parse_args()

    asyncio.run(get_audio_file_from_server(args.host, args.port, args.text, args.model, args.speaker))
