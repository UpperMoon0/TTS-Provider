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
        self.generator = TTSGenerator()
        self.files_dir = Path("generated_files")
        os.makedirs(self.files_dir, exist_ok=True)
        
    def run(self):
        """Run the WebSocket server"""
        self.logger.info(f"Starting TTS WebSocket server on {self.host}:{self.port}")
        asyncio.run(self.start_server())
    
    async def start_server(self):
        """Start the WebSocket server"""
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
    
    async def handle_client(self, websocket, path):
        """Handle client connections"""
        try:
            request_str = await websocket.recv()
            self.logger.info(f"Received request from client")
            
            try:
                request = json.loads(request_str)
                text = request.get("text", "")
                speaker = request.get("speaker", 0)
                
                # Ensure we extract the sample rate from the request
                sample_rate = request.get("sample_rate")
                
                if not sample_rate:
                    sample_rate = 24000  # Default sample rate
                
                # Get response mode - stream (default) or file
                response_mode = request.get("response_mode", "stream")
                
                self.logger.info(f"Request: text='{text}', speaker={speaker}, sample_rate={sample_rate}, response_mode={response_mode}")
                
                # First response: Model loading status
                if not self.generator.is_ready():
                    await websocket.send(json.dumps({
                        "status": "loading",
                        "message": "Model is still loading, please wait"
                    }))
                
                # Generate the audio
                try:
                    # Pass the sample_rate parameter explicitly to the generator
                    audio_bytes = await self.generator.generate_speech(
                        text=text, 
                        speaker=speaker,
                        sample_rate=sample_rate  # Explicitly pass sample_rate
                    )
                    
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
                        
                        # Send the audio data
                        await websocket.send(audio_bytes)
                        self.logger.info(f"Successfully sent {len(audio_bytes)} bytes of audio data")
                    
                except Exception as e:
                    error_msg = str(e)
                    self.logger.error(f"Error generating audio: {error_msg}")
                    await websocket.send(json.dumps({
                        "status": "error",
                        "message": f"Failed to generate speech: {error_msg}"
                    }))
            
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
