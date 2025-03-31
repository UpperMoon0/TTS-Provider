import io
import logging
import asyncio
import edge_tts
from typing import Dict, List, Tuple
from .base_model import BaseTTSModel

class EdgeTTSModel(BaseTTSModel):
    """Microsoft Edge TTS model implementation"""
    
    # List of Edge TTS voices with mapping to our numeric speaker IDs
    VOICE_MAPPING = {
        0: "en-US-GuyNeural",           # Default male voice
        1: "en-US-JennyNeural",         # Default female voice
        2: "en-US-AriaNeural",          # Alternative female voice
        3: "en-GB-RyanNeural",          # British male voice
        4: "en-GB-SoniaNeural",         # British female voice 
        5: "en-AU-WilliamNeural",       # Australian male voice
        6: "en-AU-NatashaNeural",       # Australian female voice
        7: "en-CA-LiamNeural",          # Canadian male voice
        8: "en-CA-ClaraNeural",         # Canadian female voice
        9: "en-IN-PrabhatNeural",       # Indian male voice
        10: "en-IN-NeerjaNeural",       # Indian female voice
        # Add more voices as needed
    }
    
    def __init__(self):
        """Initialize the Edge TTS model"""
        self.logger = logging.getLogger("EdgeTTSModel")
        self.ready = True  # Edge TTS doesn't require loading
        self.sample_rate = 24000  # Default sample rate
        self._voices = None
    
    async def load(self) -> bool:
        """Load the Edge TTS model - this is a no-op since Edge TTS doesn't require loading"""
        try:
            # Just check that we can list voices
            await self._get_available_voices()
            self.ready = True
            self.logger.info("Edge TTS is available")
            return True
        except Exception as e:
            self.logger.error(f"Failed to check Edge TTS availability: {str(e)}")
            self.ready = False
            return False
    
    async def _get_available_voices(self) -> List[Dict]:
        """Get the available voices from Edge TTS"""
        if self._voices is None:
            try:
                self._voices = await edge_tts.list_voices()
                self.logger.info(f"Retrieved {len(self._voices)} voices from Edge TTS")
            except Exception as e:
                self.logger.error(f"Failed to list Edge TTS voices: {str(e)}")
                self._voices = []
        return self._voices
    
    def is_ready(self) -> bool:
        """Check if the model is ready"""
        return self.ready
    
    def get_sample_rate(self) -> int:
        """Get the sample rate of the generated audio"""
        return self.sample_rate
    
    @property
    def model_name(self) -> str:
        """Get the name of the model"""
        return "Microsoft Edge TTS"
    
    @property
    def supported_speakers(self) -> dict:
        """Get the supported speakers"""
        return {
            0: "US Male (Guy)",
            1: "US Female (Jenny)",
            2: "US Female (Aria)",
            3: "UK Male (Ryan)",
            4: "UK Female (Sonia)",
            5: "Australian Male (William)",
            6: "Australian Female (Natasha)",
            7: "Canadian Male (Liam)",
            8: "Canadian Female (Clara)",
            9: "Indian Male (Prabhat)",
            10: "Indian Female (Neerja)",
        }
    
    async def generate_speech(self, text: str, speaker: int = 0, **kwargs) -> bytes:
        """Generate speech from text using Edge TTS"""
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Get voice name from speaker ID, fallback to default if not found
        voice = self.VOICE_MAPPING.get(speaker, self.VOICE_MAPPING[0])
        
        # Optional parameters
        rate = kwargs.get("rate", "+0%")  # Can be e.g. "+10%" or "-5%"
        volume = kwargs.get("volume", "+0%")  # Can be e.g. "+10%" or "-5%"
        pitch = kwargs.get("pitch", "+0Hz")  # Can be e.g. "+10Hz" or "-5Hz"
        
        self.logger.info(f"Generating speech using Edge TTS:")
        self.logger.info(f" - Text: '{text[:100]}...' ({len(text)} chars)")
        self.logger.info(f" - Voice: {voice} (speaker ID: {speaker})")
        self.logger.info(f" - Rate: {rate}, Volume: {volume}, Pitch: {pitch}")
        
        try:
            # Create communication object
            communicate = edge_tts.Communicate(text, voice)
            
            # Set additional properties if provided
            if rate != "+0%":
                communicate.rate = rate
            if volume != "+0%":
                communicate.volume = volume
            if pitch != "+0Hz":
                communicate.pitch = pitch
            
            # Use a memory buffer to store the audio
            buffer = io.BytesIO()
            
            # Generate the audio
            self.logger.info("Generating audio with Edge TTS...")
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buffer.write(chunk["data"])
            
            # Get the audio data
            buffer.seek(0)
            audio_data = buffer.read()
            
            self.logger.info(f"Generated {len(audio_data)/1024:.1f} KB of audio with Edge TTS")
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Error generating speech with Edge TTS: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to generate speech with Edge TTS: {str(e)}") 