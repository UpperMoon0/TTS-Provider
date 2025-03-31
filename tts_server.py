import json
import asyncio
import logging
import traceback
import os
import uuid
from pathlib import Path
import websockets
from tts_generator import TTSGenerator

class TTSServer:
    """WebSocket server for text-to-speech conversion"""
    
    def __init__(self, host="0.0.0.0", port=8765):
        """Initialize the TTS server with host and port"""
        self.host = host
        self.port = port
        self.logger = logging.getLogger("TTSServer")
        
        # Get the default model from environment or use the default
        default_model = os.environ.get("TTS_MODEL", "sesame")
        self.logger.info(f"Using default TTS model: {default_model}")
        
        self.generator = TTSGenerator(model_name=default_model)
        self.files_dir = Path("generated_files")
        os.makedirs(self.files_dir, exist_ok=True)
        self.model_loading = False
        self.model_loaded = False
        self.request_queue = asyncio.Queue()
        self.queue_processor_task = None
        
    def run(self):
        """Run the WebSocket server"""
        self.logger.info(f"Starting TTS WebSocket server on {self.host}:{self.port}")
        asyncio.run(self.start_server())
    
    async def start_server(self):
        """Start the WebSocket server"""
        # Start the queue processor task if the model is already loaded
        if self.model_loaded and self.queue_processor_task is None:
            self.queue_processor_task = asyncio.create_task(self.process_queued_requests())
            
        async with websockets.serve(
            self.handle_client, 
            self.host, 
            self.port,
            ping_interval=None,
            ping_timeout=None,
            max_size=None, 
            max_queue=None
        ):
            self.logger.info(f"Server started on {self.host}:{self.port}")
            await asyncio.Future()  # Run forever
    
    async def preload_model(self):
        """Preload the TTS model to avoid delays on first request"""
        if self.model_loaded or self.model_loading:
            self.logger.info("Model already loaded or loading in progress")
            return
        
        self.model_loading = True
        try:
            # Load the model
            await asyncio.to_thread(self.generator.load_model)
            self.model_loaded = self.generator.is_ready()
            
            if self.model_loaded:
                self.logger.info("Model preloaded successfully")
                # Start processing any queued requests
                self.queue_processor_task = asyncio.create_task(self.process_queued_requests())
            else:
                self.logger.error("Failed to preload model")
        except Exception as e:
            self.logger.error(f"Error preloading model: {str(e)}")
        finally:
            self.model_loading = False
    
    async def process_queued_requests(self):
        """Process requests from the queue once the model is loaded"""
        self.logger.info("Starting to process queued requests")
        while True:
            try:
                # Get a request from the queue
                client_data = await self.request_queue.get()
                websocket, request_data = client_data
                
                # Process the request
                await self.process_request(websocket, request_data)
                
                # Mark the task as done
                self.request_queue.task_done()
            except Exception as e:
                self.logger.error(f"Error processing queued request: {str(e)}")
                
    async def handle_info_request(self, websocket):
        """Handle a request for server information"""
        from tts_generator import TTSGenerator
        
        info = {
            "status": "success",
            "server_version": "1.1.0",
            "current_model": self.generator.get_model_info(),
            "available_models": TTSGenerator.list_available_models(),
            "queue_size": self.request_queue.qsize(),
            "model_loaded": self.generator.is_ready()
        }
        
        self.logger.info("Sending server information to client")
        await websocket.send(json.dumps(info))
    
    async def handle_client(self, websocket, path):
        """Handle client connections"""
        try:
            request_str = await websocket.recv()
            self.logger.info(f"Received request from client")
            
            try:
                request = json.loads(request_str)
                
                # Check for special commands
                command = request.get("command", "")
                
                if command == "info":
                    # Return server information
                    await self.handle_info_request(websocket)
                    return
                
                # If model is not ready, queue the request and inform the client
                if not self.generator.is_ready():
                    # If model is not loading yet, start loading it
                    if not self.model_loading and not self.model_loaded:
                        # Start loading the model in the background
                        asyncio.create_task(self.preload_model())
                    
                    # Inform client that their request is queued
                    await websocket.send(json.dumps({
                        "status": "queued",
                        "message": "Model is loading, your request has been queued",
                        "queue_position": self.request_queue.qsize() + 1
                    }))
                    
                    # Add to queue
                    await self.request_queue.put((websocket, request))
                    
                    # Don't close the connection, the queue processor will handle it
                    # Keep the connection open until the model is ready
                    try:
                        # Wait for a message that will never come (client disconnection or timeout)
                        await websocket.recv()
                    except websockets.exceptions.ConnectionClosed:
                        self.logger.info("Client disconnected while waiting in queue")
                else:
                    # Model is ready, process directly
                    await self.process_request(websocket, request)
                
            except json.JSONDecodeError:
                self.logger.error("Invalid JSON in request")
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": "Invalid request format: expected JSON"
                }))
                
        except websockets.exceptions.ConnectionClosedOK:
            self.logger.info("Client disconnected")
        except Exception as e:
            self.logger.error(f"Error handling client: {str(e)}")
            self.logger.debug(traceback.format_exc())
    
    async def process_request(self, websocket, request):
        """Process a TTS request once the model is ready"""
        try:
            text = request.get("text", "")
            speaker = request.get("speaker", 0)
            sample_rate = request.get("sample_rate", 24000)
            response_mode = request.get("response_mode", "stream")
            max_audio_length_ms = request.get("max_audio_length_ms", 30000)  # Default to 30 seconds if not specified
            model_type = request.get("model", None)  # Optional model selection
            
            # Additional model-specific parameters
            extra_params = {}
            
            # For Edge TTS, support voice styling parameters
            if request.get("rate"):
                extra_params["rate"] = request.get("rate")
            if request.get("volume"):
                extra_params["volume"] = request.get("volume")
            if request.get("pitch"):
                extra_params["pitch"] = request.get("pitch")
            
            text_length = len(text)
            text_preview = text[:100] + "..." if len(text) > 100 else text
            
            self.logger.info(f"Processing request:")
            self.logger.info(f" - Text length: {text_length} chars")
            self.logger.info(f" - Text preview: '{text_preview}'")
            self.logger.info(f" - Speaker: {speaker}")
            self.logger.info(f" - Sample rate: {sample_rate}")
            self.logger.info(f" - Response mode: {response_mode}")
            self.logger.info(f" - Max audio length: {max_audio_length_ms} ms")
            if model_type:
                self.logger.info(f" - Requested model: {model_type}")
            
            # Generate the audio
            try:
                # If a specific model was requested, create a new generator with that model
                model_generator = self.generator
                if model_type and model_type.lower() != os.environ.get("TTS_MODEL", "sesame"):
                    from tts_generator import TTSGenerator
                    model_generator = TTSGenerator(model_name=model_type)
                    self.logger.info(f"Using requested model: {model_type}")
                
                # Pass the sample_rate parameter explicitly to the generator
                self.logger.info(f"Calling generator with text of {text_length} chars...")
                start_time = asyncio.get_event_loop().time()
                
                audio_bytes = await model_generator.generate_speech(
                    text=text, 
                    speaker=speaker,
                    sample_rate=sample_rate,
                    max_audio_length_ms=max_audio_length_ms,
                    **extra_params
                )
                
                end_time = asyncio.get_event_loop().time()
                generation_time = end_time - start_time
                
                audio_size_kb = len(audio_bytes) / 1024
                self.logger.info(f"Generated {audio_size_kb:.1f} KB of audio in {generation_time:.1f} seconds")
                self.logger.info(f"Audio bytes length: {len(audio_bytes)}")
                
                if response_mode == "file":
                    # Generate a unique filename
                    filename = f"tts_{uuid.uuid4()}.wav"
                    filepath = self.files_dir / filename
                    
                    # Save the audio to a file
                    with open(filepath, "wb") as f:
                        f.write(audio_bytes)
                    
                    # Send metadata with file information
                    metadata = {
                        "status": "success",
                        "response_mode": "file",
                        "length_bytes": len(audio_bytes),
                        "sample_rate": sample_rate,
                        "format": "wav",
                        "filename": str(filename),
                        "filepath": str(filepath)
                    }
                    await websocket.send(json.dumps(metadata))
                    self.logger.info(f"Saved audio to file: {filepath}")
                    
                else:  # Default stream mode
                    # Send metadata
                    metadata = {
                        "status": "success",
                        "response_mode": "stream",
                        "length_bytes": len(audio_bytes),
                        "sample_rate": sample_rate,
                        "format": "wav"
                    }
                    await websocket.send(json.dumps(metadata))
                    
                    # Check if we need to chunk the response (over ~1MB)
                    MAX_CHUNK_SIZE = 800000  # ~800KB to stay safely under 1MB limit
                    if len(audio_bytes) > MAX_CHUNK_SIZE:
                        self.logger.info(f"Audio response is {len(audio_bytes)} bytes, chunking into smaller fragments")
                        
                        # Send data in chunks
                        total_chunks = (len(audio_bytes) + MAX_CHUNK_SIZE - 1) // MAX_CHUNK_SIZE
                        for i in range(0, len(audio_bytes), MAX_CHUNK_SIZE):
                            chunk = audio_bytes[i:i + MAX_CHUNK_SIZE]
                            await websocket.send(chunk)
                            self.logger.debug(f"Sent chunk {(i // MAX_CHUNK_SIZE) + 1}/{total_chunks} ({len(chunk)} bytes)")
                        
                        self.logger.info(f"Successfully sent {len(audio_bytes)} bytes of audio data in {total_chunks} chunks")
                    else:
                        # Send the audio data in one go
                        await websocket.send(audio_bytes)
                        self.logger.info(f"Successfully sent {len(audio_bytes)} bytes of audio data")
                
            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"Error generating audio: {error_msg}")
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": f"Failed to generate speech: {error_msg}"
                }))
                
        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            try:
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": f"Internal server error: {str(e)}"
                }))
            except:
                pass
