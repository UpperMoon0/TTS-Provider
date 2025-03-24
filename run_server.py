#!/usr/bin/env python3
import sys
import logging
import argparse
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
    return parser.parse_args()

def main():
    """Main entry point for the TTS WebSocket Server."""
    logger = setup_logging()
    args = parse_arguments()
    
    try:
        logger.info(f"Starting TTS WebSocket Server on {args.host}:{args.port}")
        server = TTSServer(host=args.host, port=args.port)
        server.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error running server: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())