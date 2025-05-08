#!/usr/bin/env python3
"""
Main script to run the TTS server
"""

import os
import logging
import argparse
import logging
import os
from dotenv import load_dotenv

from model_loader import ModelLoader # Import ModelLoader
from tts_server import TTSServer

def setup_logging():
    """Set up logging for the server"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main function to run the TTS server"""
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger("TTS-Server-Main")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the TTS server")
    parser.add_argument("--host", default=os.environ.get("TTS_HOST", "0.0.0.0"), help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=int(os.environ.get("TTS_PORT", 9000)), help="Port to bind the server to")
    parser.add_argument("--model", default=os.environ.get("TTS_MODEL", "edge"), help="Default TTS model to use (e.g., 'sesame', 'edge')")
    parser.add_argument("--preload", type=bool, default=False, help="Whether to preload the model at startup (default: False)")
    args = parser.parse_args()
    
    # Log the configuration
    logger.info(f"Starting TTS server:")
    logger.info(f" - Host: {args.host}")
    logger.info(f" - Port: {args.port}")
    logger.info(f" - Default model: {args.model}")
    logger.info(f" - Preload model: {args.preload}")
    
    # Set the model in the environment for the server to use
    if args.model:
        os.environ["TTS_MODEL"] = args.model
    
    # Create and run the server
    server = TTSServer(host=args.host, port=args.port)
    
    # Preload the model if requested (only beneficial for Edge TTS which is lightweight)
    if args.preload:
        logger.info("Preloading the model...")
        import asyncio
        asyncio.run(server.preload_model())
    else:
        logger.info("Lazy loading enabled - models will be loaded on first request")
    
    # Run the server
    server.run()

if __name__ == "__main__":
    main()
