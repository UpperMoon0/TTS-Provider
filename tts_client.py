import os
import json
import asyncio
import logging
import shutil
import websockets
from pathlib import Path

class TTSClient:
    """Client for connecting to the TTS server and generating speech."""
    
    def __init__(self, host='localhost', port=9000, timeout=60):
        """Initialize the TTS client.
        
        Args:
            host (str): The host of the TTS server
            port (int): The port of the TTS server
            timeout (int): The timeout for operations in seconds
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.uri = f"ws://{host}:{port}"
        self.websocket = None
        self.logger = logging.getLogger("TTS-Client")
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging for the client."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def is_connected(self):
        """Check if the client is connected to the server.
        
        Returns:
            bool: True if connected, False otherwise
        """
        return self.websocket is not None and not self.websocket.closed
    
    async def connect(self):
        """Connect to the TTS server.
        
        Returns:
            bool: True if connected successfully, False otherwise
        """
        if self.is_connected():
            self.logger.warning("Already connected to the server")
            return True
        
        try:
            self.logger.info(f"Connecting to TTS server at {self.uri}")
            self.websocket = await asyncio.wait_for(
                websockets.connect(
                    self.uri,
                    max_size=10*1024*1024,  # 10 MB max message size
                    ping_interval=None,
                    open_timeout=self.timeout
                ),
                timeout=self.timeout
            )
            self.logger.info("Connected to the TTS server")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to the TTS server: {str(e)}")
            return False
    
    async def disconnect(self):
        """Disconnect from the TTS server."""
        if self.is_connected():
            self.logger.info("Disconnecting from the TTS server")
            await self.websocket.close()
            self.websocket = None
        else:
            self.logger.warning("Not connected to the server")
    
    async def generate_speech(self, text, output_path, speaker=0, sample_rate=24000, 
                           response_mode="stream", model=None, **kwargs):
        """Generate speech from the given text and save it to the output path.
        
        Args:
            text (str): The text to convert to speech
            output_path (str): The path where the generated audio will be saved
            speaker (int): The speaker ID to use
            sample_rate (int): The sample rate of the generated audio
            response_mode (str): The response mode, either "stream" or "file"
            model (str, optional): The model to use (e.g., "sesame", "edge")
            **kwargs: Additional model-specific parameters:
                      - For Edge TTS: rate, volume, pitch
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected():
            self.logger.error("Not connected to the server")
            raise ConnectionError("Not connected to the server")
        
        request = {
            "text": text,
            "speaker": speaker,
            "sample_rate": sample_rate,
            "response_mode": response_mode
        }
        
        # Add model if specified
        if model:
            request["model_type"] = model
            
        # Add additional parameters if provided
        extra_params = {}
        for param in ["rate", "volume", "pitch"]:
            if param in kwargs:
                extra_params[param] = kwargs[param]
                
        if extra_params:
            request["extra_params"] = extra_params
        
        try:
            self.logger.info(f"Sending TTS request: {json.dumps(request)}")
            await self.websocket.send(json.dumps(request))
            
            # Wait for metadata response
            self.logger.info("Waiting for metadata...")
            metadata_str = await asyncio.wait_for(self.websocket.recv(), timeout=self.timeout)
            metadata = json.loads(metadata_str)
            self.logger.info(f"Received metadata: {json.dumps(metadata)}")
            
            # Handle model loading or queued status
            status = metadata.get("status")
            if status == "loading":
                self.logger.info("Model is loading, waiting for completion...")
                metadata_str = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=self.timeout
                )
                metadata = json.loads(metadata_str)
                self.logger.info(f"Updated metadata: {json.dumps(metadata)}")
            elif status == "queued":
                queue_position = metadata.get("queue_position", "unknown")
                self.logger.info(f"Request queued (position: {queue_position}), waiting for processing...")
                
                # Wait for the server to process our request from the queue
                metadata_str = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=self.timeout
                )
                metadata = json.loads(metadata_str)
                self.logger.info(f"Request processed from queue: {json.dumps(metadata)}")
            
            # Check for successful status
            if metadata.get("status") != "success":
                error_msg = metadata.get("message", "Unknown error")
                self.logger.error(f"TTS generation failed: {error_msg}")
                return False
            
            # Create output directory if needed
            output_path_obj = Path(output_path)
            os.makedirs(output_path_obj.parent, exist_ok=True)
            
            # Process based on response mode
            received_mode = metadata.get("response_mode", "stream")
            
            if received_mode == "file":
                # File mode - copy the server-generated file to the output path
                server_filepath = metadata.get("filepath")
                if not server_filepath:
                    self.logger.error("Server did not provide a filepath")
                    return False
                
                self.logger.info(f"Server generated file at {server_filepath}")
                self.logger.info(f"Copying file to {output_path}")
                
                # Copy the file - note we're not using websockets for binary data transfer
                shutil.copyfile(server_filepath, output_path)
                self.logger.info(f"Audio file saved to {output_path}")
                
                return True
            else:
                # Stream mode - get audio data from websocket and save it
                self.logger.info("Waiting for audio stream data...")
                audio_data = await asyncio.wait_for(self.websocket.recv(), timeout=self.timeout)
                self.logger.info(f"Received {len(audio_data)} bytes of audio data")
                
                # Save audio data to file
                with open(output_path, "wb") as f:
                    f.write(audio_data)
                
                self.logger.info(f"Audio saved to {output_path}")
                return True
                
        except Exception as e:
            self.logger.error(f"TTS generation failed: {str(e)}")
            raise
    
    async def get_server_info(self):
        """Get information about the TTS server, including available models.
        
        Returns:
            dict: Server information including available models, or None if failed
        """
        if not self.is_connected():
            self.logger.error("Not connected to the server")
            raise ConnectionError("Not connected to the server")
            
        request = {
            "command": "info"
        }
        
        try:
            self.logger.info("Requesting server information")
            await self.websocket.send(json.dumps(request))
            
            # Wait for response
            response_str = await asyncio.wait_for(self.websocket.recv(), timeout=self.timeout)
            info = json.loads(response_str)
            self.logger.info(f"Received server information: {json.dumps(info)}")
            
            return info
        except Exception as e:
            self.logger.error(f"Failed to get server information: {str(e)}")
            raise