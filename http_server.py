from fastapi import FastAPI
import uvicorn
import logging

def start_http_server(host, port, tts_service):
    """Start the HTTP server for the TTS provider"""
    app = FastAPI()
    logger = logging.getLogger("TTS-HTTP-Server")
    
    # Import and include the HTTP routes
    from api.http_routes import create_http_routes
    http_router = create_http_routes(tts_service)
    app.include_router(http_router)
    
    logger.info(f"Starting HTTP server on {host}:{port}")
    
    # Run the server
    uvicorn.run(app, host=host, port=port, log_level="info")