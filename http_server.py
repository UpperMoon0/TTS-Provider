#!/usr/bin/env python3
"""
HTTP Server for TTS Provider monitoring and API endpoints
"""

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from core.config import Config
from services.tts_service import TTSService
from api.http_routes import router as http_router, set_tts_service

def create_app(tts_service_instance=None):
    """Create and configure the FastAPI application."""
    # Create FastAPI app
    app = FastAPI(
        title="TTS-Provider API",
        description="HTTP API for TTS Provider monitoring and management",
        version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify allowed origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Set the TTS service instance for the routes
    if tts_service_instance:
        set_tts_service(tts_service_instance)
    
    # Include the HTTP routes
    app.include_router(http_router)
    
    return app

# Global app instance
app = None

def start_http_server(host="0.0.0.0", port=8001, tts_service_instance=None):
    """Start the HTTP server for monitoring endpoints."""
    global app
    logger = logging.getLogger("TTS-HTTP-Server")
    logger.info(f"Starting HTTP server on {host}:{port}")
    
    # Create the app with the TTS service instance
    app = create_app(tts_service_instance)
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False
    )

if __name__ == "__main__":
    # Start HTTP server when run directly
    # Create a TTS service instance for direct execution
    tts_service = TTSService()
    start_http_server(tts_service_instance=tts_service)