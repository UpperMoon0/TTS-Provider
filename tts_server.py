import asyncio
import json
import logging
import threading
import time
import traceback
import websockets
from queue import Queue

from tts_generator import TTSGenerator

class TTSServer:
    """WebSocket server for Text-to-Speech services."""
    
    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.logger = logging.getLogger("TTSServer")
        self.tts_generator = TTSGenerator()
        self.server = None
        self.model_loaded = False
        self.model_loading = False
        self.model_load_timeout = 900  # 15 minutes
        self.pending_requests = []
        self.main_event_loop = None
        self.model_loaded_callback_queue = Queue()
        
    async def handle_client(self, websocket):
        """Handle WebSocket client connection."""
        client_id = id(websocket)
        self.logger.info(f"Client {client_id} connected")
        
        try:
            async for message in websocket:
                try:
                    # Parse the incoming JSON message
                    data = json.loads(message)
                    
                    # Extract the text and optional speaker ID
                    text = data.get("text", "")
                    speaker = data.get("speaker", 0)
                    
                    if not text:
                        await self._send_error(websocket, "No text provided")
                        continue
                    
                    self.logger.info(f"Client {client_id} requested TTS: speaker={speaker}, text='{text}'")
                    
                    # Check if model is loaded
                    if not self.model_loaded:
                        if not self.model_loading:
                            # Start loading the model
                            self.logger.info("Model not loaded, starting loading process")
                            await self._start_model_loading()
                        
                        # Inform client that model is loading
                        await websocket.send(json.dumps({
                            "status": "loading",
                            "message": "TTS model is still loading, request will be processed when ready"
                        }))
                        
                        # Add to pending requests
                        self.pending_requests.append((websocket, text, speaker))
                        continue
                    
                    # Process the request immediately if model is loaded
                    await self._process_tts_request(websocket, text, speaker)
                    
                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid JSON format")
                except Exception as e:
                    self.logger.error(f"Error processing request: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    await self._send_error(websocket, f"Internal server error: {str(e)}")
        except websockets.exceptions.ConnectionClosed as e:
            self.logger.info(f"Client {client_id} disconnected: {e.code} {e.reason}")
        except Exception as e:
            self.logger.error(f"Unhandled exception with client {client_id}: {str(e)}")
            self.logger.error(traceback.format_exc())
        finally:
            # Remove any pending requests from this client
            self.pending_requests = [req for req in self.pending_requests if req[0] != websocket]
            self.logger.info(f"Client {client_id} connection closed")
    
    async def _send_error(self, websocket, message):
        """Send an error message to the client."""
        try:
            await websocket.send(json.dumps({
                "status": "error",
                "message": message
            }))
        except Exception as e:
            self.logger.error(f"Failed to send error message: {str(e)}")
    
    async def _process_tts_request(self, websocket, text, speaker):
        """Process TTS request and send the audio to the client."""
        try:
            # Generate audio with timeout
            self.logger.info(f"Generating audio for: '{text[:50]}...' (truncated)")
            start_time = time.time()
            
            # Run in executor to avoid blocking the event loop
            audio_bytes = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.tts_generator.generate_wav_bytes(text, speaker, timeout=60)
            )
            
            elapsed = time.time() - start_time
            self.logger.info(f"Audio generation took {elapsed:.2f} seconds, size: {len(audio_bytes)} bytes")
            
            # Send metadata
            metadata = {
                "status": "success",
                "message": "Audio generated successfully",
                "format": "wav",
                "sample_rate": self.tts_generator.sample_rate,
                "length_bytes": len(audio_bytes)
            }
            
            self.logger.info(f"Sending metadata: {json.dumps(metadata)}")
            await websocket.send(json.dumps(metadata))
            
            # Send binary audio data
            self.logger.info(f"Sending audio data ({len(audio_bytes)} bytes)...")
            await websocket.send(audio_bytes)
            self.logger.info("Audio data sent successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating audio: {str(e)}")
            self.logger.error(traceback.format_exc())
            await self._send_error(websocket, f"Error generating audio: {str(e)}")
    
    async def _start_model_loading(self):
        """Start loading the TTS model in a background thread."""
        if self.model_loading:
            return
        
        self.model_loading = True
        self.main_event_loop = asyncio.get_event_loop()
        
        def _load_model():
            try:
                self.logger.info("Starting model loading...")
                start_time = time.time()
                success = self.tts_generator.load_model()
                elapsed = time.time() - start_time
                
                if success:
                    self.logger.info(f"Model loaded successfully in {elapsed:.2f} seconds")
                    self.model_loaded = True
                    
                    # Signal the main thread that model loading is complete
                    # Instead of directly calling the coroutine, we queue it for the main thread
                    self.model_loaded_callback_queue.put(True)
                else:
                    self.logger.error(f"Failed to load model after {elapsed:.2f} seconds")
                    self.model_loading = False
            except Exception as e:
                self.logger.error(f"Error loading model: {str(e)}")
                self.logger.error(traceback.format_exc())
                self.model_loading = False
        
        # Start model loading in a separate thread
        thread = threading.Thread(target=_load_model)
        thread.daemon = True
        thread.start()
        
        # Create a task to check the model loaded queue
        asyncio.create_task(self._check_model_loaded_queue())
        
        # Set a timeout for model loading
        asyncio.create_task(self._check_model_loading_timeout(thread))
    
    async def _check_model_loaded_queue(self):
        """Check if the model has been loaded and process pending requests."""
        while True:
            # Check if there's a message in the queue without blocking
            try:
                if not self.model_loaded_callback_queue.empty():
                    # Get the message (we don't actually use the value)
                    self.model_loaded_callback_queue.get()
                    
                    # Process pending requests
                    await self._process_pending_requests()
                    break
                await asyncio.sleep(0.1)  # Short sleep to avoid CPU spinning
            except Exception as e:
                self.logger.error(f"Error checking model loaded queue: {str(e)}")
                self.logger.error(traceback.format_exc())
                await asyncio.sleep(1)  # Longer sleep on error
    
    async def _check_model_loading_timeout(self, thread):
        """Check if model loading has timed out."""
        start_time = time.time()
        while thread.is_alive() and time.time() - start_time < self.model_load_timeout:
            await asyncio.sleep(1)
        
        if thread.is_alive() and not self.model_loaded:
            self.logger.error(f"Model loading timed out after {self.model_load_timeout} seconds")
            self.model_loading = False
            
            # Notify all pending clients
            for websocket, _, _ in self.pending_requests:
                try:
                    await websocket.send(json.dumps({
                        "status": "error",
                        "message": "Model loading timed out"
                    }))
                except Exception:
                    pass
            
            self.pending_requests = []
    
    async def _process_pending_requests(self):
        """Process all pending requests after model is loaded."""
        self.logger.info(f"Processing {len(self.pending_requests)} pending requests")
        
        pending = self.pending_requests
        self.pending_requests = []
        
        for websocket, text, speaker in pending:
            try:
                await self._process_tts_request(websocket, text, speaker)
            except Exception as e:
                self.logger.error(f"Error processing pending request: {str(e)}")
                try:
                    await self._send_error(websocket, f"Error processing request: {str(e)}")
                except Exception:
                    pass
    
    async def start_server(self):
        """Start the WebSocket server."""
        self.server = await websockets.serve(
            self.handle_client, 
            self.host, 
            self.port,
            ping_interval=30,  # Send ping every 30 seconds
            ping_timeout=10,   # Wait 10 seconds for pong response
            max_size=10*1024*1024  # 10MB max message size for long texts
        )
        self.logger.info(f"WebSocket server started at ws://{self.host}:{self.port}")
        
        # Start loading the model
        await self._start_model_loading()
    
    def run(self):
        """Run the server (blocking call)."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.start_server())
        loop.run_forever()
    
    async def stop(self):
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.logger.info("WebSocket server stopped")
