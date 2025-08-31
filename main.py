#!/usr/bin/env python3
"""
Main application module for TTS Provider
"""

import os
import logging
import argparse
import asyncio
import threading
from dotenv import load_dotenv

from core.config import Config
from services.tts_service import TTSService
from api.websocket_routes import WebSocketRoutes
from http_server import start_http_server

def setup_logging():
    """Set up logging for the server"""
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
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
    parser.add_argument("--host", default=Config.HOST, help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=Config.PORT, help="Port to bind the server to")
    # --model argument removed as per user request
    args = parser.parse_args()
    
    # Log the configuration
    logger.info(f"Starting TTS server:")
    logger.info(f" - Host: {args.host}")
    logger.info(f" - Port: {args.port}")
    # The default model is 'edge' if not specified by the client (see TTSService).
    logger.info(f" - Default model: 'edge' (if not specified by client)")
    
    # The TTSService will use its own internal default ('edge') if no model is specified in a request.
    # No need to set os.environ["TTS_MODEL"] from a command-line arg here.
    
    # Create TTS service
    global tts_service
    tts_service = TTSService()
    
    # Start HTTP server in a separate thread, passing the TTS service instance
    logger.info("Starting HTTP server...")
    http_port = int(os.getenv("TTS_HTTP_PORT", "8001"))
    http_thread = threading.Thread(
        target=start_http_server,
        args=(args.host, http_port, tts_service),  # Pass the TTS service instance
        daemon=True
    )
    http_thread.start()
    
    # Start WebSocket server
    logger.info("Starting WebSocket server...")
    websocket_server = WebSocketRoutes(tts_service, host=args.host, port=args.port)
    
    logger.info("Lazy loading enabled - models will be loaded on first request")
    
    # Run the WebSocket server
    websocket_server.run()

if __name__ == "__main__":
    main()