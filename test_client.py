#!/usr/bin/env python
import asyncio
import json
import websockets
import argparse
import wave
import os
import sys

async def test_tts_client(uri, text, speaker=0, output_file="output.wav"):
    """
    Test client for the TTS WebSocket server.
    
    Args:
        uri: WebSocket URI to connect to (e.g., ws://localhost:8765)
        text: Text to convert to speech
        speaker: Speaker ID (0 for male, 1 for female)
        output_file: File to save the audio to
    """
    print(f"Connecting to {uri}...")
    
    async with websockets.connect(uri) as websocket:
        print(f"Connected. Requesting TTS for: '{text}'")
        
        # Create request
        request = {
            "text": text,
            "speaker": speaker
        }
        
        # Send request
        await websocket.send(json.dumps(request))
        print("Request sent, waiting for response...")
        
        # Receive JSON metadata
        response = await websocket.recv()
        metadata = json.loads(response)
        print(f"Received metadata: {metadata}")
        
        if "error" in metadata:
            print(f"Error: {metadata['error']}")
            return False
            
        if metadata.get("status") != "success":
            print(f"Unexpected response: {metadata}")
            return False
        
        # Receive audio data
        audio_data = await websocket.recv()
        print(f"Received {len(audio_data)} bytes of audio data")
        
        # Save to file
        with open(output_file, "wb") as f:
            f.write(audio_data)
        
        print(f"Audio saved to {output_file}")
        return True

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="TTS WebSocket Client Test")
    parser.add_argument("--host", default="localhost", help="TTS server hostname")
    parser.add_argument("--port", type=int, default=8765, help="TTS server port")
    parser.add_argument("--text", default="Hello, this is a test of the TTS WebSocket server.", help="Text to convert to speech")
    parser.add_argument("--speaker", type=int, default=0, choices=[0, 1], help="Speaker ID (0 for male, 1 for female)")
    parser.add_argument("--output", default="output.wav", help="Output WAV file")
    return parser.parse_args()

def main():
    """Main entry point for the TTS test client"""
    args = parse_arguments()
    
    # Build WebSocket URI
    uri = f"ws://{args.host}:{args.port}"
    
    try:
        # Run the async client
        result = asyncio.run(test_tts_client(uri, args.text, args.speaker, args.output))
        return 0 if result else 1
        
    except KeyboardInterrupt:
        print("Client stopped by user")
    except Exception as e:
        print(f"Error running client: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())