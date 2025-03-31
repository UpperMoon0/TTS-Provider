#!/usr/bin/env python3
import os
import asyncio
import argparse
import logging
from pathlib import Path
import sys

# Add the parent directory to the path so we can import the client
sys.path.insert(0, str(Path(__file__).parent.parent))
from tts_client import TTSClient

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Client-Example")

async def main():
    parser = argparse.ArgumentParser(description="TTS Client Example")
    parser.add_argument("--host", default="localhost", help="TTS server host")
    parser.add_argument("--port", type=int, default=9000, help="TTS server port")
    parser.add_argument("--text", default="Hello, this is a test. Generating speech using multiple models.", 
                        help="Text to convert to speech")
    parser.add_argument("--output-dir", default="./output", help="Directory to save the output files")
    parser.add_argument("--model", default=None, help="Model to use (sesame, edge), if not specified will try both")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create client and connect
    client = TTSClient(host=args.host, port=args.port)
    
    try:
        # Connect to the server
        connected = await client.connect()
        if not connected:
            logger.error("Could not connect to the server. Exiting.")
            return
        
        # Get server information
        logger.info("Requesting server information...")
        server_info = await client.get_server_info()
        logger.info(f"Server version: {server_info.get('server_version')}")
        logger.info(f"Current model: {server_info.get('current_model')}")
        logger.info(f"Available models: {server_info.get('available_models')}")
        
        models_to_try = []
        if args.model:
            # Use specified model
            models_to_try = [args.model]
        else:
            # Try all available models
            models_to_try = server_info.get('available_models', ['sesame'])
        
        # Generate speech with each model
        for model in models_to_try:
            output_path = os.path.join(args.output_dir, f"output_{model}.wav")
            logger.info(f"Generating speech using {model} model...")
            
            extra_params = {}
            if model == "edge":
                # Example of Edge TTS specific parameters
                extra_params = {
                    "rate": "+10%",  # Speak 10% faster
                    "volume": "+20%",  # 20% louder
                    "pitch": "-5%"  # Slightly deeper voice
                }
            
            await client.generate_speech(
                text=args.text,
                output_path=output_path,
                model=model,
                **extra_params
            )
            
            logger.info(f"Speech generated and saved to {output_path}")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    finally:
        # Disconnect from the server
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main()) 