import asyncio
import json
import logging
import os
import websockets
import threading
from tts_generator import TTSGenerator
from dotenv import load_dotenv

class TTSServer:
    """
    WebSocket server for the TTS service.
    Handles client connections and processes TTS requests.
    """
    
    def __init__(self, host="0.0.0.0", port=8765):
        """
        Initialize the TTS Server
        
        Args:
            host: Host to bind the server to
            port: Port to listen on
        """
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
        
        # Lock for model loading
        self.model_load_lock = asyncio.Lock()
        
        # Event to signal when model is loaded
        self.model_loaded_event = asyncio.Event()
        
        self.logger.info(f"TTS Server initialized on {host}:{port}")
    
    async def load_model(self):
        """Load the TTS model asynchronously"""
        # Use a lock to prevent multiple simultaneous loading attempts
        async with self.model_load_lock:
            if not self.model_loaded:
                self.logger.info("Starting to load TTS model...")
                
                # Create a future for the loading task
                loop = asyncio.get_event_loop()
                
                # Run model loading in a thread to not block the event loop
                self.model_loaded = await loop.run_in_executor(
                    None, self.tts_generator.load_model
                )
                
                if self.model_loaded:
                    self.logger.info("TTS model loaded successfully")
                    # Set the event to signal model is loaded
                    self.model_loaded_event.set()
                else:
                    self.logger.error("Failed to load TTS model")
    
    async def handle_client(self, websocket):
        """
        Handle a client connection
        
        Args:
            websocket: WebSocket connection object
        """
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
        """
        Process a message from a client
        
        Args:
            websocket: WebSocket connection object
            message: Message from the client
            client_id: Client ID for logging
        """
        try:
            # Parse JSON request
            request = json.loads(message)
            
            # Validate request
            if 'text' not in request:
                await websocket.send(json.dumps({'error': 'Missing required field: text'}))
                return
            
            # Get parameters
            text = request['text']
            speaker = int(request.get('speaker', 0))
            
            # Log the request
            self.logger.info(f"Client {client_id} requested TTS: speaker={speaker}, text='{text}'")
            
            # Wait for model to be loaded if it's not already
            if not self.model_loaded:
                await websocket.send(json.dumps({
                    'status': 'waiting',
                    'message': 'TTS model is still loading, please wait'
                }))
                await self.model_loaded_event.wait()
            
            # Generate audio
            try:
                # Generate WAV bytes
                wav_bytes = self.tts_generator.generate_wav_bytes(text, speaker)
                
                # Send success message
                await websocket.send(json.dumps({
                    'status': 'success',
                    'message': 'Audio generated successfully',
                    'format': 'wav',
                    'sample_rate': self.tts_generator.sample_rate,
                    'length_bytes': len(wav_bytes)
                }))
                
                # Send the audio data
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
        
        # Create the websocket server
        async with websockets.serve(self.handle_client, self.host, self.port):
            # Wait for model to load in parallel with serving requests
            self.logger.info(f"WebSocket server is running on {self.host}:{self.port}")
            self.logger.info("Server is waiting for model to finish loading before accepting requests")
            
            # Keep the server running indefinitely
            await asyncio.Future()
            
    def run(self):
        """Run the server (blocking call)"""
        asyncio.run(self.start())