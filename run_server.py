#!/usr/bin/env python3
import sys
import logging
import argparse
import asyncio
from tts_server import TTSServer

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("TTS-Service")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TTS WebSocket Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    parser.add_argument("--preload-model", action="store_true", default=True, 
                        help="Preload the model at startup (default: True)")
    return parser.parse_args()

async def load_model(server, logger):
    """Load the TTS model."""
    logger.info("Preloading TTS model (this may take a while)...")
    await server.preload_model()
    logger.info("TTS model loaded successfully")

def main():
    """Main entry point for the TTS WebSocket Server."""
    logger = setup_logging()
    args = parse_arguments()
    
    try:
        logger.info(f"Starting TTS WebSocket Server on {args.host}:{args.port}")
        server = TTSServer(host=args.host, port=args.port)
        
        # Load model if requested
        if args.preload_model:
            # We need to use asyncio to call the async preload_model method
            asyncio.run(load_model(server, logger))
        
        server.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error running server: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())