import io
import logging
import os
import torchaudio
from typing import Optional, Dict, Any

from tts_models.factory import TTSModelFactory
from tts_models.base_model import BaseTTSModel

class TTSGenerator:
    """Text-to-speech generator supporting multiple TTS backends"""
    
    # Default model to use
    DEFAULT_MODEL = "sesame"
    
    def __init__(self, model_name: str = None, max_audio_length_ms: int = None):
        """
        Initialize the TTS generator
        
        Args:
            model_name: Name of the model to use (default: sesame)
            max_audio_length_ms: Maximum audio length in milliseconds
        """
        self.logger = logging.getLogger("TTSGenerator")
        self.ready = False
        self.max_audio_length_ms = max_audio_length_ms or 120000  # Default to 120 seconds if None
        
        # Create the model
        model_name = model_name or os.environ.get("TTS_MODEL", self.DEFAULT_MODEL)
        self.logger.info(f"Initializing TTSGenerator with model: {model_name}")
        
        self.model = TTSModelFactory.create_model(model_name)
        if self.model is None:
            self.logger.error(f"Unknown model: {model_name}, falling back to {self.DEFAULT_MODEL}")
            self.model = TTSModelFactory.create_model(self.DEFAULT_MODEL)
            
        self.sample_rate = self.model.get_sample_rate()
        self.logger.info(f"Using model: {self.model.model_name}, sample rate: {self.sample_rate}")
    
    def load_model(self) -> bool:
        """Load the TTS model"""
        self.logger.info(f"Loading TTS model: {self.model.model_name}...")
        
        import asyncio
        result = asyncio.run(self.model.load())
        self.ready = result
        
        if self.ready:
            self.logger.info(f"TTS model loaded successfully")
            return True
        else:
            self.logger.error(f"Failed to load TTS model")
            return False
    
    def is_ready(self) -> bool:
        """Check if the model is ready"""
        return self.ready or self.model.is_ready()
    
    async def generate_speech(self, text: str, speaker: int = 0, sample_rate: Optional[int] = None, 
                            max_audio_length_ms: Optional[int] = None, **kwargs) -> bytes:
        """
        Generate speech asynchronously
        
        Args:
            text: Text to convert to speech
            speaker: Speaker ID
            sample_rate: Sample rate of the generated audio
            max_audio_length_ms: Maximum audio length in milliseconds
            **kwargs: Additional model-specific parameters
            
        Returns:
            Audio bytes in WAV format
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Check if model is ready
        if not self.is_ready():
            self.logger.info("Model not ready, loading...")
            if not await self._async_load_model():
                raise RuntimeError("Model failed to load. Check logs for details.")
        
        text_length = len(text)
        self.logger.info(f"Generating speech:")
        self.logger.info(f" - Model: {self.model.model_name}")
        self.logger.info(f" - Text length: {text_length} chars")
        self.logger.info(f" - Speaker: {speaker}")
        
        # Use provided parameters or defaults
        max_audio_length = max_audio_length_ms or self.max_audio_length_ms
        params = {
            "max_audio_length_ms": max_audio_length,
            **kwargs
        }
        
        # Generate speech using the model
        try:
            audio_bytes = await self.model.generate_speech(text, speaker, **params)
            
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
    
    async def _async_load_model(self) -> bool:
        """Load the model asynchronously"""
        return await self.model.load()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if not self.model:
            return {}
            
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