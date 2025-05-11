import logging
import os
import torch # Added for GPU check
from typing import Optional, Dict, Any

from tts_models.factory import TTSModelFactory

class TTSGenerator:
    """Text-to-speech generator supporting multiple TTS backends"""
    
    # Default model to use
    DEFAULT_MODEL = "edge"  # Changed default to edge since it's lightweight
    
    # Dictionary to cache model instances by name
    _model_cache = {}
    
    def __init__(self, model_name: str = None, max_audio_length_ms: int = None):
        """
        Initialize the TTS generator
        
        Args:
            model_name: Name of the model to use (default: edge)
            max_audio_length_ms: Maximum audio length in milliseconds
        """
        self.logger = logging.getLogger("TTSGenerator")
        self.ready = False
        self.max_audio_length_ms = max_audio_length_ms or 120000  # Default to 120 seconds if None
        
        # Determine which model to use
        self.model_name = model_name or os.environ.get("TTS_MODEL", self.DEFAULT_MODEL)
        self.logger.info(f"Initializing TTSGenerator with model: {self.model_name}")
        
        # Initialize with Edge TTS by default as it's lightweight
        if self.model_name.lower() in ["edge", "edge-tts"]:
            self._initialize_model(self.model_name)
        else:
            # For Sesame/other models, just store the name but don't initialize
            self.model = None
            self.sample_rate = 24000  # Default sample rate
    
    def _initialize_model(self, model_name: str) -> None:
        """Initialize a specific model instance"""
        self.model = TTSModelFactory.create_model(model_name)
        if self.model is None:
            self.logger.error(f"Unknown model: {model_name}, falling back to {self.DEFAULT_MODEL}")
            self.model = TTSModelFactory.create_model(self.DEFAULT_MODEL)
            
        self.sample_rate = self.model.get_sample_rate()
        self.logger.info(f"Using model: {self.model.model_name}, sample rate: {self.sample_rate}")
    
    def load_model(self) -> bool:
        """Load the TTS model"""
        # Ensure model is initialized before loading
        if self.model is None:
            self._initialize_model(self.model_name)
            
        self.logger.info(f"Loading TTS model: {self.model.model_name}...")
        
        # Log GPU availability for local models
        if self.model_name.lower() not in ["edge", "edge-tts"]:
            if torch.cuda.is_available():
                self.logger.info(f"CUDA GPU is available. Device count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    self.logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                self.logger.info(f"  Current CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            else:
                self.logger.info("CUDA GPU is not available. TTS will run on CPU.")
        
        import asyncio
        result = asyncio.run(self.model.load()) # Pass websocket if needed by model.load()
        self.ready = result
        
        if self.ready:
            self.logger.info(f"TTS model loaded successfully")
            return True
        else:
            self.logger.error(f"Failed to load TTS model")
            return False
    
    def is_ready(self) -> bool:
        """Check if the model is ready"""
        # Edge TTS is always ready immediately
        if self.model_name.lower() in ["edge", "edge-tts"]:
            return True
            
        # For other models, check if they're loaded
        return self.model is not None and (self.ready or self.model.is_ready())
    
    async def generate_speech(self, text: str, speaker: int = 0, lang: str = "en-US", 
                            sample_rate: Optional[int] = None, websocket=None, **kwargs) -> bytes: # Added websocket
        """
        Generate speech asynchronously
        
        Args:
            text: Text to convert to speech
            speaker: Speaker ID
            lang: Language code (e.g., "en-US", "ja-JP")
            sample_rate: Sample rate of the generated audio
            # max_audio_length_ms: Maximum audio length in milliseconds (Removed)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Audio bytes in WAV format
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Check if we're using a specific model for this request
        requested_model = kwargs.get("model", self.model_name).lower()
        
        # If a specific model is requested but not initialized, initialize it now
        if requested_model != self.model_name or self.model is None:
            self.logger.info(f"Switching to model: {requested_model}")
            self.model_name = requested_model
            self._initialize_model(requested_model)
        
        # Check if model is ready
        if not self.is_ready():
            self.logger.info("Model not ready, loading...")
            # Pass websocket to _async_load_model
            if not await self._async_load_model(websocket=websocket): # Pass websocket here
                raise RuntimeError("Model failed to load. Check logs for details.")
        
        text_length = len(text)
        self.logger.info(f"Generating speech:")
        self.logger.info(f" - Model: {self.model.model_name}")
        self.logger.info(f" - Text length: {text_length} chars")
        self.logger.info(f" - Speaker: {speaker}")
        self.logger.info(f" - Language: {lang}")
        
        # Use provided parameters or defaults
        # max_audio_length = max_audio_length_ms or self.max_audio_length_ms # Removed
        params = {
            # "max_audio_length_ms": max_audio_length, # Removed
            **kwargs
        }
        
        # Generate speech using the model
        try:
            # Pass lang and websocket parameters to the model's generate_speech method
            audio_bytes = await self.model.generate_speech(text, speaker, lang=lang, websocket=websocket, **params) # Pass websocket here
            
            # Success
            wav_size_kb = len(audio_bytes) / 1024
            self.logger.info(f"Generated {wav_size_kb:.1f} KB of audio")
            return audio_bytes
            
        except Exception as e:
            self.logger.error(f"Error generating TTS audio: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Propagate the error
            raise RuntimeError(f"Failed to generate speech: {str(e)}")
    
    async def _async_load_model(self, websocket=None) -> bool: # Added websocket parameter
        """Load the model asynchronously"""
        # Ensure the model is initialized
        if self.model is None:
            self._initialize_model(self.model_name)
            
        # Pass websocket to the model's load method
        result = await self.model.load(websocket=websocket)
        self.ready = result
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if not self.model:
            # If model isn't initialized yet, initialize it now
            self._initialize_model(self.model_name)
            
        return {
            "name": self.model.model_name,
            "ready": self.is_ready(),
            "sample_rate": self.sample_rate,
            "speakers": self.model.supported_speakers
        }
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Dict]:
        """List all available models and their information"""
        return TTSModelFactory.get_model_info()
