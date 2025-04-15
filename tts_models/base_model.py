from abc import ABC, abstractmethod

class BaseTTSModel(ABC):
    """Base class for all TTS models"""
    
    @abstractmethod
    async def generate_speech(self, text: str, speaker: int = 0, lang: str = "en-US", **kwargs) -> bytes:
        """
        Generate speech from text
        
        Args:
            text: Text to convert to speech
            speaker: Speaker ID
            lang: Language code (e.g., "en-US", "ja-JP")
            # max_audio_length_ms: Removed
            **kwargs: Additional model-specific parameters
            
        Returns:
            Audio bytes in WAV format
        """
        pass
    
    @abstractmethod
    def get_sample_rate(self) -> int:
        """
        Get the sample rate of the generated audio
        
        Returns:
            Sample rate in Hz
        """
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """
        Check if the model is ready to generate speech
        
        Returns:
            True if the model is ready, False otherwise
        """
        pass
    
    @abstractmethod
    async def load(self) -> bool:
        """
        Load the model
        
        Returns:
            True if the model was loaded successfully, False otherwise
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Get the name of the model
        
        Returns:
            Model name
        """
        pass
    
    @property
    @abstractmethod
    def supported_speakers(self) -> dict:
        """
        Get the supported speakers
        
        Returns:
            Dict mapping speaker IDs to descriptions
        """
        pass
