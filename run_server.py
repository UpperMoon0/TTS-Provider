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
    # --model argument removed as per user request
    args = parser.parse_args()
    
    # Log the configuration
    logger.info(f"Starting TTS server:")
    logger.info(f" - Host: {args.host}")
    logger.info(f" - Port: {args.port}")
    # The default model is 'edge' if not specified by the client (see TTSServer).
    logger.info(f" - Default model: 'edge' (if not specified by client)")
    
    # The TTSServer will use its own internal default ('edge') if no model is specified in a request.
    # No need to set os.environ["TTS_MODEL"] from a command-line arg here.
    
    # Create and run the server
    server = TTSServer(host=args.host, port=args.port)
    
    logger.info("Lazy loading enabled - models will be loaded on first request")
    
    # Run the server
    server.run()

if __name__ == "__main__":
    main()
