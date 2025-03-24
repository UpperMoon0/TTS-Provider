#!/usr/bin/env python
"""
Test client for the TTS WebSocket Server
"""

import sys
import json
import time
import asyncio
import logging
import argparse
import socket
import websockets

def setup_logging():
    """Configure logging for the test client"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(stream=sys.stdout)
        ]
    )
    return logging.getLogger("TTS-TestClient")

def is_port_in_use(port, host='127.0.0.1'):
    """Check if a port is already in use."""
    try:
        # Create a socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Try to connect to the port
            s.settimeout(1)
            result = s.connect_ex((host, port))
            return result == 0  # If result is 0, port is in use
    except socket.error:
        return False

# Removed start_server_process function as server will be started separately

async def wait_for_server(host, port, timeout=300):  # Increased timeout to 5 minutes
    """Wait for the server to be ready to accept connections"""
    logger = logging.getLogger("TTS-TestClient")
    
    if host == '0.0.0.0':
        connect_host = '127.0.0.1'  # Use localhost to connect to 0.0.0.0
    else:
        connect_host = host
        
    uri = f"ws://{connect_host}:{port}"
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            async with websockets.connect(uri, max_size=10*1024*1024, close_timeout=5, ping_timeout=10) as websocket:
                # Try a simple ping to see if the server is ready
                pong = await websocket.ping()
                await asyncio.wait_for(pong, timeout=5)  # Increased timeout
                logger.info(f"Server at {uri} is ready")
                return True
        except (websockets.exceptions.ConnectionClosedError, 
                websockets.exceptions.InvalidStatusCode,
                asyncio.TimeoutError,
                ConnectionRefusedError):
            logger.debug(f"Server at {uri} not ready yet, retrying...")
            await asyncio.sleep(1)
    
    logger.error(f"Timed out waiting for server at {uri}")
    return False

async def run_test(host="localhost", port=8765, text=None, speaker=0, output_file=None, start_server=False, debug=False):
    """Run a test request to the TTS server"""
    logger = setup_logging()
    
    if not text:
        text = "This is a test of the text-to-speech server. If you can hear this, the server is working correctly."
    
    # Removed server starting code
    
    # Determine the host to connect to (127.0.0.1 for 0.0.0.0)
    connect_host = '127.0.0.1' if host == '0.0.0.0' else host
    uri = f"ws://{connect_host}:{port}"
    
    # Wait for the server to be ready
    if not await wait_for_server(host, port, timeout=300):  # Increased timeout to 5 minutes
        logger.error("Server is not responding, test failed")
        return False
        
    logger.info(f"Connecting to TTS server at {uri}")
    
    try:
        async with websockets.connect(uri, max_size=10*1024*1024) as websocket:
            # Send request
            request = {
                "text": text,
                "speaker": speaker
            }
            
            logger.info(f"Sending request: {json.dumps(request)}")
            send_time = time.time()
            await websocket.send(json.dumps(request))
            logger.info("Request sent")
            
            # Wait for metadata
            logger.info("Waiting for metadata response...")
            metadata_str = await websocket.recv()
            metadata_time = time.time()
            
            logger.info(f"Received metadata in {metadata_time - send_time:.2f}s")
            try:
                metadata = json.loads(metadata_str)
                logger.info(f"Metadata: {json.dumps(metadata)}")
                
                if metadata.get("status") == "loading":
                    logger.info("Model is still loading, waiting for completion...")
                    # Wait for the actual response after model loads
                    metadata_str = await websocket.recv()
                    metadata = json.loads(metadata_str)
                    logger.info(f"Updated metadata: {json.dumps(metadata)}")
                
                if metadata.get("status") != "success":
                    logger.error(f"Error from server: {metadata.get('message', 'Unknown error')}")
                    return False
                
                # Get binary audio data
                logger.info(f"Waiting for {metadata.get('length_bytes', 'unknown')} bytes of audio data...")
                audio_data = await websocket.recv()
                receive_time = time.time()
                
                logger.info(f"Received {len(audio_data)} bytes of audio data in {receive_time - metadata_time:.2f}s")
                logger.info(f"Total request time: {receive_time - send_time:.2f}s")
                
                # Save to file if requested
                if output_file:
                    with open(output_file, "wb") as f:
                        f.write(audio_data)
                    logger.info(f"Audio saved to {output_file}")
                
                return True
                
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in metadata: {metadata_str}")
                return False
                
    except websockets.exceptions.ConnectionClosed as e:
        logger.error(f"WebSocket connection closed unexpectedly: {e.code} {e.reason}")
        return False
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        return False
    # Removed server stopping code since server is started separately

def main():
    """Main entry point for the test client"""
    parser = argparse.ArgumentParser(description="TTS WebSocket Test Client")
    parser.add_argument("--host", default="localhost", help="Server hostname")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--text", help="Text to convert to speech")
    parser.add_argument("--speaker", type=int, default=0, choices=[0, 1], help="Speaker ID (0=male, 1=female)")
    parser.add_argument("--output", help="Output WAV file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Set default output file if not specified
    if not args.output:
        args.output = f"tts_output_{int(time.time())}.wav"
    
    # Run the test
    result = asyncio.run(run_test(
        host=args.host,
        port=args.port,
        text=args.text,
        speaker=args.speaker,
        output_file=args.output,
        start_server=False,  # Never start the server
        debug=args.debug
    ))
    
    # Exit with appropriate status code
    sys.exit(0 if result else 1)

if __name__ == "__main__":
    main()
