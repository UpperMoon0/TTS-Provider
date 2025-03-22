import os
import sys
import torch
import torchaudio
import logging
import argparse
import asyncio
import json
import websockets
from pathlib import Path
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

def verify_setup():
    """Verify that the required files and directories exist"""
    csm_path = os.path.join(os.path.dirname(__file__), 'csm')
    if not os.path.exists(csm_path):
        print("Error: CSM repository not found. Please clone it first:")
        print("git clone https://github.com/SesameAILabs/csm.git")
        return False
    return True

class TTSServer:
    """WebSocket server for TTS service using CSM-1B model."""
    
    def __init__(self, host="0.0.0.0", port=8765):
        self.logger = logging.getLogger("TTSServer")
        self.host = host
        self.port = port
        
        # Load environment variables
        load_dotenv()
        
        # Get HF settings from environment
        self.hf_home = os.getenv("HF_HOME", "D:/Dev/Models/huggingface")
        self.hf_token = os.getenv("HF_TOKEN")
        
        # Initialize TTS Generator
        self.tts_generator = TTSGenerator(
            hf_home=self.hf_home,
            hf_token=self.hf_token,
            max_audio_length_ms=30000
        )
        
        # Flag to track if model is loaded
        self.model_loaded = False
        self.model_load_lock = asyncio.Lock()
        self.model_loaded_event = asyncio.Event()
        
        self.logger.info(f"TTS Server initialized on {host}:{port}")
    
    async def load_model(self):
        """Load the TTS model asynchronously"""
        async with self.model_load_lock:
            if not self.model_loaded:
                self.logger.info("Starting to load TTS model...")
                loop = asyncio.get_event_loop()
                self.model_loaded = await loop.run_in_executor(None, self.tts_generator.load_model)
                
                if self.model_loaded:
                    self.logger.info("TTS model loaded successfully")
                    self.model_loaded_event.set()
                else:
                    self.logger.error("Failed to load TTS model")
    
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
        
        # Start loading the model immediately
        self.logger.info("Starting model loading process")
        load_task = asyncio.create_task(self.load_model())
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            self.logger.info(f"WebSocket server is running on {self.host}:{self.port}")
            self.logger.info("Server is waiting for model to finish loading before accepting requests")
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