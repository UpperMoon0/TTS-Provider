import json
import asyncio
import logging
import traceback
import os
from pathlib import Path
import websockets

class TTSServer:
    """WebSocket server for text-to-speech conversion"""
    
    # Speaker mapping table for cross-model compatibility
    # When using integer speaker IDs, this helps map them consistently
    # regardless of which model is used
    SPEAKER_MAPPING = {
        # Generic mappings that work across models - Zonos will use its own speaker IDs based on filenames
        0: {"description": "Default Male Voice", "edge": 0},   # Default Male (US Guy)
        1: {"description": "Default Female Voice", "edge": 1}, # Default Female (US Jenny)
        2: {"description": "Alternative Male Voice", "edge": 2},  # Alternative Male (US Davis)
        3: {"description": "Alternative Female Voice", "edge": 4}, # Alternative Female (UK Sonia)
    }
    
    def __init__(self, host="0.0.0.0", port=9000):
        """Initialize the TTS server with host and port"""
        self.host = host
        self.port = port
        self.logger = logging.getLogger("TTSServer")
        
        # Default model is hardcoded to "edge" if not specified in request
        # TTS_MODEL environment variable is no longer used for default model selection
        self.initial_default_model = "edge" 
        self.logger.info(f"Initial default TTS model if not specified by client: {self.initial_default_model}")
        self.generator = None
        # Use absolute path to ensure files are saved in the TTS-Provider directory
        self.files_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "generated_files"
        self.model_loading = False
        self.model_loaded = False
        self.request_queue = asyncio.Queue()
        self.queue_processor_task = None
        
    def map_speaker_id(self, speaker_id: int, model_type: str) -> int:
        """Map a generic speaker ID to a model-specific speaker ID"""
        # Default to the same speaker ID if no mapping exists
        if speaker_id not in self.SPEAKER_MAPPING:
            return speaker_id
            
        # Get the mapping for this speaker
        mapping = self.SPEAKER_MAPPING[speaker_id]
        
        # Map to the appropriate model's speaker ID
        if model_type.lower() in ["edge", "edge-tts"]: # Adjusted from elif to if
            return mapping.get("edge", 0)
        # For Zonos or other models not explicitly mapped here,
        # the original speaker_id is typically used directly by the model.
        # (e.g., Zonos maps integer IDs to its reference audio files like 0.wav, 1.wav)
        else:
            # For unknown or unmapped models (like Zonos), return the original ID
            self.logger.debug(f"Speaker ID {speaker_id} for model '{model_type}' not in explicit mapping, using original ID.")
            return speaker_id
    
    def run(self):
        """Run the WebSocket server"""
        self.logger.info(f"Starting TTS WebSocket server on {self.host}:{self.port}")
        asyncio.run(self.start_server())
    
    async def start_server(self):
        """Start the WebSocket server"""
        # Lazy import to delay loading the model until needed
        from tts_generator import TTSGenerator
        
        # Initialize the generator here to avoid loading the model too early
        if self.generator is None:
            # TTSGenerator is initialized with the hardcoded default model "edge".
            # It will be used if a client request doesn't specify a model.
            self.generator = TTSGenerator(model_name=self.initial_default_model)
            
        # Start the queue processor task if the model is already loaded
        if self.model_loaded and self.queue_processor_task is None:
            self.queue_processor_task = asyncio.create_task(self.process_queued_requests())
            
        async with websockets.serve(
            self.handle_client, 
            self.host, 
            self.port,
            ping_interval=20,      # Send ping frames every 20 seconds
            ping_timeout=30,       # Wait 30 seconds for pong response
            close_timeout=10,      # Wait 10 seconds for close frame
            max_size=10 * 1024 * 1024,  # Increase max message size to 10MB
            max_queue=100          # Allow up to 100 pending messages
        ):
            self.logger.info(f"Server started on {self.host}:{self.port}")
            await asyncio.Future()  # Run forever
    
    async def preload_model(self, websocket=None): # Added websocket parameter
        """Preload the TTS model to avoid delays on first request.
        Can optionally ping the provided websocket during loading."""
        if self.model_loaded or self.model_loading:
            self.logger.info("Model already loaded or loading in progress")
            return
        
        self.model_loading = True
        try:
            # Load the model, passing the websocket object
            self.logger.info(f"Calling self.generator.load_model in a thread, passing websocket: {websocket is not None}")
            await asyncio.to_thread(self.generator.load_model, websocket=websocket) # Pass websocket
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
        
        # Get model info and add speaker mapping information
        model_info = self.generator.get_model_info()
        
        # Get available models
        available_models = TTSGenerator.list_available_models()
        
        # Add speaker mapping information for client reference
        speaker_mapping = {}
        for speaker_id, mapping in self.SPEAKER_MAPPING.items():
            speaker_mapping[speaker_id] = mapping["description"]
        
        info = {
            "status": "success",
            "server_version": "1.1.0",
            "current_model": model_info,
            "available_models": available_models,
            "queue_size": self.request_queue.qsize(),
            "model_loaded": self.generator.is_ready(),
            "speaker_mapping": speaker_mapping
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
                        # Start loading the model in the background, passing the client's websocket
                        self.logger.info(f"Model not ready, creating preload_model task for websocket: {websocket.remote_address}")
                        asyncio.create_task(self.preload_model(websocket=websocket)) # Pass websocket here
                    
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
            # response_mode = request.get("response_mode", "stream") # Removed as per user request
            # max_audio_length_ms = request.get("max_audio_length_ms", 30000) # Removed parameter
            model_type = request.get("model", self.generator.model_name)  # Optional model selection
            lang = request.get("lang", "en-US") # Add language parameter, default to en-US
            
            # Map the speaker ID to the appropriate model-specific ID
            mapped_speaker = self.map_speaker_id(speaker, model_type)
            
            # Additional model-specific parameters
            extra_params = {}
            
            # If a specific model was requested, add it to the parameters
            if model_type:
                extra_params["model"] = model_type
            
            text_length = len(text)
            text_preview = text[:100] + "..." if len(text) > 100 else text
            
            self.logger.info(f"Processing request:")
            self.logger.info(f" - Text length: {text_length} chars")
            self.logger.info(f" - Text preview: '{text_preview}'")
            self.logger.info(f" - Original speaker: {speaker}, Mapped speaker: {mapped_speaker}")
            self.logger.info(f" - Language: {lang}")
            self.logger.info(f" - Sample rate: {sample_rate}")
            # self.logger.info(f" - Response mode: {response_mode}") # Removed log
            # self.logger.info(f" - Max audio length: {max_audio_length_ms} ms") # Removed log
            if model_type:
                self.logger.info(f" - Requested model: {model_type}")
            
            # Generate the audio
            try:
                # Pass the model_type through extra_params to support dynamic model loading
                self.logger.info(f"Calling generator with text of {text_length} chars...")
                start_time = asyncio.get_event_loop().time()
                
                audio_bytes = await self.generator.generate_speech(
                    text=text,
                    speaker=mapped_speaker,  # Use the mapped speaker ID
                    lang=lang,               # Pass the language
                    sample_rate=sample_rate,
                    websocket=websocket,     # Pass websocket here
                    # max_audio_length_ms=max_audio_length_ms, # Removed parameter
                    **extra_params
                )
                
                end_time = asyncio.get_event_loop().time()
                generation_time = end_time - start_time
                
                audio_size_kb = len(audio_bytes) / 1024
                self.logger.info(f"Generated {audio_size_kb:.1f} KB of audio in {generation_time:.1f} seconds")
                self.logger.info(f"Audio bytes length: {len(audio_bytes)}")
                
                # Always stream the audio
                # Send metadata
                metadata = {
                    "status": "success",
                    # "response_mode": "stream", # Removed as it's always stream
                    "length_bytes": len(audio_bytes),
                    "sample_rate": sample_rate,
                    "format": "wav"
                }
                await websocket.send(json.dumps(metadata))
                
                # Adding delay to prevent connection issues
                await asyncio.sleep(0.5)
                
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
                        # Add a small delay between chunks
                        await asyncio.sleep(0.1)
                    self.logger.info(f"Successfully sent {len(audio_bytes)} bytes of audio data in {total_chunks} chunks")
                else:
                    # Send the audio data in one go
                    await websocket.send(audio_bytes)
                    self.logger.info(f"Successfully sent {len(audio_bytes)} bytes of audio data")
                
                # Add a delay before potentially closing the connection
                await asyncio.sleep(0.5)
                
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
