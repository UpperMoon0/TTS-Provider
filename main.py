import os
import sys
import logging
import argparse
import asyncio
import json
import websockets
from dotenv import load_dotenv
from tts_generator import TTSGenerator

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

class TTSServer:
    """WebSocket server for TTS service using CSM-1B model."""
    
    def __init__(self, host="0.0.0.0", port=8765):
        self.logger = logging.getLogger("TTSServer")
        self.host = host
        self.port = port
        
        # Load environment variables
        load_dotenv()
        
        # Initialize TTS Generator - no need to pass HF settings as they're read from env
        self.tts_generator = TTSGenerator(max_audio_length_ms=30000)
        
        # Flag to track if model is loaded
        self.model_loaded = False
        self.model_load_lock = asyncio.Lock()
        self.model_loaded_event = asyncio.Event()
        
        self.logger.info(f"TTS Server initialized on {host}:{port}")
    
    async def load_model(self):
        """Load the TTS model asynchronously and wait for completion"""
        async with self.model_load_lock:
            if not self.model_loaded:
                self.logger.info("Starting to load TTS model...")
                loop = asyncio.get_event_loop()
                
                # Run model loading in a thread pool and wait for it to complete
                try:
                    self.model_loaded = await loop.run_in_executor(None, self.tts_generator.load_model)
                    
                    if self.model_loaded:
                        self.logger.info("TTS model loaded successfully")
                        self.model_loaded_event.set()
                    else:
                        self.logger.error("Failed to load TTS model")
                        raise RuntimeError("Model loading failed")
                except Exception as e:
                    self.logger.error(f"Error loading model: {str(e)}")
                    raise
    
    async def handle_client(self, websocket):
        """Handle a client connection"""
        client_id = id(websocket)
        self.logger.info(f"Client {client_id} connected")
        
        try:
            # Make sure client waits until model is loaded
            if not self.model_loaded:
                await websocket.send(json.dumps({
                    'status': 'loading',
                    'message': 'TTS model is still loading, request will be processed when ready'
                }))
                await self.model_loaded_event.wait()
            
            async for message in websocket:
                await self.process_message(websocket, message, client_id)
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            self.logger.error(f"Error handling client {client_id}: {str(e)}")
    
    async def process_message(self, websocket, message, client_id):
        """Process a message from a client"""
        try:
            request = json.loads(message)
            
            if 'text' not in request:
                await websocket.send(json.dumps({'error': 'Missing required field: text'}))
                return
            
            text = request['text']
            speaker = int(request.get('speaker', 0))
            
            self.logger.info(f"Client {client_id} requested TTS: speaker={speaker}, text='{text}'")
            
            try:
                wav_bytes = self.tts_generator.generate_wav_bytes(text, speaker)
                
                await websocket.send(json.dumps({
                    'status': 'success',
                    'message': 'Audio generated successfully',
                    'format': 'wav',
                    'sample_rate': self.tts_generator.sample_rate,
                    'length_bytes': len(wav_bytes)
                }))
                
                await websocket.send(wav_bytes)
                self.logger.info(f"Sent {len(wav_bytes)} bytes of audio data to client {client_id}")
                
            except Exception as e:
                self.logger.error(f"Error generating audio: {str(e)}")
                await websocket.send(json.dumps({
                    'status': 'error',
                    'message': f'Error generating audio: {str(e)}'
                }))
                
        except json.JSONDecodeError:
            await websocket.send(json.dumps({'error': 'Invalid JSON format'}))
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            await websocket.send(json.dumps({'error': f'Server error: {str(e)}'}))
    
    async def start(self):
        """Start the WebSocket server"""
        self.logger.info(f"Starting TTS WebSocket server on {self.host}:{self.port}")
        
        # Load the model first and wait for completion
        self.logger.info("Loading TTS model before starting server...")
        await self.load_model()
        
        if not self.model_loaded:
            self.logger.error("Failed to load TTS model. Server cannot start.")
            return
        
        self.logger.info("Model loaded successfully, starting WebSocket server...")
        
        # Create the websocket server only after model is loaded
        async with websockets.serve(self.handle_client, self.host, self.port):
            self.logger.info(f"WebSocket server is running on {self.host}:{self.port}")
            await asyncio.Future()  # Keep the server running indefinitely
    
    def run(self):
        """Run the server (blocking call)"""
        asyncio.run(self.start())

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="TTS WebSocket Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    return parser.parse_args()

def main():
    """Main entry point for the TTS WebSocket Server"""
    
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