#!/usr/bin/env python
import os
import sys
import logging
import argparse
from tts_server import TTSServer

def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(stream=sys.stdout)
        ]
    )
    return logging.getLogger("TTS-Service")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="TTS WebSocket Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    return parser.parse_args()

def verify_setup():
    """Verify that the required files and directories exist"""
    csm_path = os.path.join(os.path.dirname(__file__), 'csm')
    if not os.path.exists(csm_path):
        print("Error: CSM repository not found. Please clone it first:")
        print("git clone https://github.com/SesameAILabs/csm.git")
        return False
    return True

def main():
    """Main entry point for the TTS WebSocket Server"""
    # Verify setup first
    if not verify_setup():
        return 1

    # Set up logging
    logger = setup_logging()
    logger.info("Starting TTS WebSocket Server")
    
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Create and run the TTS server
        server = TTSServer(host=args.host, port=args.port)
        
        # Run the server (blocking call)
        logger.info(f"Server running at ws://{args.host}:{args.port}")
        logger.info("Model will be loaded at startup before accepting requests")
        logger.info("Press Ctrl+C to exit")
        server.run()
    
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error running server: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())