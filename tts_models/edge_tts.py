import io
import logging
import asyncio
import re
import edge_tts
from typing import Dict, List, Tuple
from .base_model import BaseTTSModel

class EdgeTTSModel(BaseTTSModel):
    """Microsoft Edge TTS model implementation"""
    
    # List of Edge TTS voices with mapping to our numeric speaker IDs
    VOICE_MAPPING = {
        0: "en-US-GuyNeural",           # Default male voice (Narrator)
        1: "en-US-JennyNeural",         # Default female voice (Jane)
        2: "en-US-DavisNeural",         # Different male voice (Valentino)
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
    
    # Fallback voices for each primary voice
    FALLBACK_VOICES = {
        "en-US-GuyNeural": ["en-US-DavisNeural", "en-GB-RyanNeural"],
        "en-US-JennyNeural": ["en-US-AriaNeural", "en-GB-SoniaNeural"],
        "en-US-DavisNeural": ["en-US-GuyNeural", "en-GB-RyanNeural"],
        "en-GB-RyanNeural": ["en-US-GuyNeural", "en-US-DavisNeural"],
        "en-GB-SoniaNeural": ["en-US-JennyNeural", "en-US-AriaNeural"]
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
            2: "US Male (Davis)",
            3: "UK Male (Ryan)",
            4: "UK Female (Sonia)",
            5: "Australian Male (William)",
            6: "Australian Female (Natasha)",
            7: "Canadian Male (Liam)",
            8: "Canadian Female (Clara)",
            9: "Indian Male (Prabhat)",
            10: "Indian Female (Neerja)",
        }
    
    def _sanitize_text(self, text: str) -> str:
        """
        Sanitize text to avoid issues with Edge TTS.
        
        Some characters or patterns can cause issues with the Edge TTS service.
        This function removes or replaces them to increase the chance of successful generation.
        """
        # Replace special quotes with standard quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("'", "'").replace("'", "'")
        
        # Replace special dashes/hyphens with standard hyphen
        text = text.replace('—', '-').replace('–', '-')
        
        # Remove any control characters
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        
        # Ensure there are no excessive spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Ensure there's at least a period at the end if there's no ending punctuation
        if text and not text[-1] in ['.', '!', '?', ':', ';']:
            text = text + '.'
            
        return text
    
    async def _verify_voice_exists(self, voice_name: str) -> bool:
        """Verify that a voice exists in the Edge TTS service"""
        voices = await self._get_available_voices()
        return any(v["ShortName"] == voice_name for v in voices)
    
    async def generate_speech(self, text: str, speaker: int = 0, **kwargs) -> bytes:
        """Generate speech from text using Edge TTS"""
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Sanitize text to avoid issues
        original_text = text
        text = self._sanitize_text(text)
        
        if text != original_text:
            self.logger.info(f"Text was sanitized for better compatibility")
            
        # Get voice name from speaker ID, fallback to default if not found
        voice = self.VOICE_MAPPING.get(speaker, self.VOICE_MAPPING[0])
        
        # Log voice without any SSML modifications
        self.logger.info(f"Generating speech using Edge TTS:")
        self.logger.info(f" - Text length: {len(text)} chars")
        self.logger.info(f" - Text preview: '{text[:100]}...' (truncated)" if len(text) > 100 else f" - Text: '{text}'")
        self.logger.info(f" - Voice: {voice} (speaker ID: {speaker})")
        self.logger.info(f" - Using default voice parameters (no SSML modifications)")
        
        # Verify the voice exists
        if not await self._verify_voice_exists(voice):
            self.logger.warning(f"Voice {voice} doesn't exist in Edge TTS. Using default voice instead.")
            voice = "en-US-GuyNeural"  # Default fallback voice
            
        # List of voices to try (primary + fallbacks)
        voices_to_try = [voice] + self.FALLBACK_VOICES.get(voice, [])
        
        errors = []
        for attempt, current_voice in enumerate(voices_to_try):
            try:
                self.logger.info(f"Attempt {attempt+1}/{len(voices_to_try)}: Using voice {current_voice}")
                
                # Create communication object - without any SSML modifications
                communicate = edge_tts.Communicate(text, current_voice)
                
                # Use a memory buffer to store the audio
                buffer = io.BytesIO()
                
                # Generate the audio
                self.logger.info("Generating audio with Edge TTS...")
                received_any_data = False
                
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        buffer.write(chunk["data"])
                        received_any_data = True
                
                # Check if we actually received audio
                buffer.seek(0)
                audio_data = buffer.read()
                
                if len(audio_data) == 0:
                    self.logger.error(f"No audio data received from Edge TTS using voice {current_voice}")
                    errors.append(f"No audio data received for voice {current_voice}")
                    continue
                
                self.logger.info(f"Generated {len(audio_data)/1024:.1f} KB of audio with Edge TTS using voice {current_voice}")
                return audio_data
                
            except Exception as e:
                self.logger.error(f"Error generating speech with Edge TTS voice {current_voice}: {str(e)}")
                errors.append(f"Voice {current_voice}: {str(e)}")
                continue
        
        # If we reach here, all attempts failed
        error_details = "\n".join(errors)
        error_message = f"Failed to generate speech with Edge TTS after trying {len(voices_to_try)} voices. Details:\n{error_details}"
        self.logger.error(error_message)
        
        # Try one last attempt with a minimal text
        try:
            self.logger.info("Attempting last-ditch generation with minimal text...")
            communicate = edge_tts.Communicate("This is a test.", "en-US-GuyNeural")
            buffer = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buffer.write(chunk["data"])
            buffer.seek(0)
            audio_data = buffer.read()
            
            if len(audio_data) > 0:
                self.logger.info("Generated fallback audio with minimal text")
                return audio_data
        except Exception as e:
            self.logger.error(f"Even minimal text generation failed: {str(e)}")
        
        # All attempts failed
        raise RuntimeError(f"Failed to generate speech with Edge TTS: {error_message}")