import io
import re
import edge_tts
from typing import Dict, List # Keep Dict and List
from .base_model import BaseTTSModel
from pydub import AudioSegment # Keep pydub import

class EdgeTTSModel(BaseTTSModel):
    """Microsoft Edge TTS model implementation"""
    
    # Language-specific voice mappings
    VOICE_MAPPINGS = {
        "en-US": {
            0: "en-US-ChristopherNeural",
            1: "en-US-JennyNeural",
            2: "en-US-DavisNeural",
            3: "en-US-AriaNeural", # Note: This was Aria, keeping it for consistency unless specified otherwise
            # Add more en-US voices if needed
        },
        "ja-JP": {
            0: "ja-JP-KeitaNeural",
            1: "ja-JP-NanamiNeural",
            # Speaker IDs 2 and 3 are not defined for ja-JP based on requirements
        }
    }
    
    DEFAULT_VOICE = "en-US-ChristopherNeural" # Default fallback if language/speaker combo fails
    
    def __init__(self):
        """Initialize the Edge TTS model"""
        super().__init__() # Call BaseTTSModel's __init__ to set up self.logger
        # self.logger is now initialized by BaseTTSModel with the class name "EdgeTTSModel"
        self.ready = True  # Edge TTS doesn't require loading
        self.sample_rate = 24000  # Default sample rate
        self._voices = None
    
    async def load(self, websocket=None) -> bool:
        """Load the Edge TTS model - this is a no-op since Edge TTS doesn't require loading"""
        
        async def _actual_load():
            try:
                # Just check that we can list voices
                await self._get_available_voices()
                self.ready = True
                self.logger.info("Edge TTS is available and ready.")
                return True
            except Exception as e:
                self.logger.error(f"Failed to check Edge TTS availability: {str(e)}")
                self.ready = False
                return False

        return await self.run_task_with_keepalive(websocket, _actual_load())
    
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
    def supported_speakers(self) -> Dict[int, str]:
        """Get the supported speakers for the default language (en-US)."""
        # Derives from supported_languages_and_voices for "en-US"
        all_langs_voices = self.supported_languages_and_voices
        return all_langs_voices.get("en-US", {})
    
    @property
    def supported_languages_and_voices(self) -> Dict[str, Dict[int, str]]:
        """Get all supported languages and the voices/speakers available for each."""
        # Transform VOICE_MAPPINGS to the required format with descriptions
        # Descriptions can be generic or derived from voice names.
        # Example: "Speaker {id} ({voice_name})"
        
        # Let's try to create more meaningful descriptions based on the voice names.
        # e.g. en-US-ChristopherNeural -> Christopher (Neural)
        
        processed_mappings = {}
        for lang_code, voices in self.VOICE_MAPPINGS.items():
            processed_mappings[lang_code] = {}
            for speaker_id, voice_name in voices.items():
                # Attempt to create a friendlier name
                # Example: "en-US-ChristopherNeural" -> "ChristopherNeural" -> "Christopher Neural"
                # Example: "ja-JP-KeitaNeural" -> "KeitaNeural" -> "Keita Neural"
                name_part = voice_name.split('-')[-1] # "ChristopherNeural" or "NanamiNeural"
                # Add a space before "Neural" if it exists and isn't the whole name
                if "Neural" in name_part and name_part != "Neural":
                    friendly_name = name_part.replace("Neural", " Neural").strip()
                else:
                    friendly_name = name_part
                
                # If the name is just "Neural" or empty after processing, use the full voice name as fallback
                if not friendly_name or friendly_name == "Neural":
                    friendly_name = voice_name

                processed_mappings[lang_code][speaker_id] = f"{friendly_name}"
        return processed_mappings

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
        
    def _get_voice_for_speaker(self, lang: str, speaker: int) -> str:
        """Get the voice name for a given language and speaker ID."""
        lang_voices = self.VOICE_MAPPINGS.get(lang)
        if not lang_voices:
            self.logger.warning(f"Language '{lang}' not configured. Falling back to default voice '{self.DEFAULT_VOICE}'.")
            return self.DEFAULT_VOICE
            
        voice = lang_voices.get(speaker)
        if not voice:
            # Fallback to speaker 0 for the requested language if the specific speaker ID doesn't exist
            fallback_voice = lang_voices.get(0)
            if fallback_voice:
                self.logger.warning(f"Speaker ID {speaker} not found for language '{lang}'. Falling back to speaker 0 ('{fallback_voice}').")
                return fallback_voice
            else:
                # If speaker 0 also doesn't exist for the language, use the absolute default
                self.logger.warning(f"Speaker ID {speaker} and fallback speaker 0 not found for language '{lang}'. Falling back to default voice '{self.DEFAULT_VOICE}'.")
                return self.DEFAULT_VOICE
        return voice

    async def generate_speech(self, text: str, speaker: int = 0, lang: str = "en-US", websocket=None, **kwargs) -> bytes:
        """Generate speech from text using Edge TTS"""
        
        async def _actual_generate():
            # max_audio_length_ms parameter removed from signature
            if not text.strip():
                raise ValueError("Text cannot be empty")
            
            # Sanitize text to avoid issues
            original_text = text
            sanitized_text = self._sanitize_text(text) # Renamed to avoid conflict
            
            if sanitized_text != original_text:
                self.logger.info(f"Text was sanitized for better compatibility")
                
            # Get the appropriate voice based on language and speaker ID
            voice = self._get_voice_for_speaker(lang, speaker)
            
            # Check if any rate, volume, or pitch parameters were passed
            if any(param in kwargs for param in ["rate", "volume", "pitch"]):
                self.logger.warning("Rate, volume, and pitch parameters are not supported in this implementation. Using default voice settings only.")
            
            # Log voice information
            self.logger.info(f"Generating speech using Edge TTS:")
            self.logger.info(f" - Text length: {len(sanitized_text)} chars")
            self.logger.info(f" - Text preview: '{sanitized_text[:100]}...' (truncated)" if len(sanitized_text) > 100 else f" - Text: '{sanitized_text}'")
            self.logger.info(f" - Language: {lang}")
            self.logger.info(f" - Selected Voice: {voice} (Original Speaker ID: {speaker})")
            self.logger.info(f" - Using default voice parameters only (SSML modifications not allowed)")
            
            # Verify the selected voice exists
            if not await self._verify_voice_exists(voice):
                self.logger.warning(f"Selected voice '{voice}' not found in Edge TTS list. Falling back to absolute default '{self.DEFAULT_VOICE}'.")
                voice = self.DEFAULT_VOICE
                # Re-verify the absolute default just in case
                if not await self._verify_voice_exists(voice):
                     raise RuntimeError(f"Default fallback voice '{self.DEFAULT_VOICE}' also not found. Cannot proceed.")
                
            # List of voices to try (selected voice)
            voices_to_try = [voice]
            
            errors = []
            for attempt, current_voice in enumerate(voices_to_try):
                try:
                    self.logger.info(f"Attempt {attempt+1}/{len(voices_to_try)}: Using voice {current_voice}")                # Create communication object with default parameters only - no SSML modifications
                    self.logger.info(" - Using default voice parameters only (no SSML modifications)")
                    
                    # Create communication object with default parameters
                    communicate = edge_tts.Communicate(
                        sanitized_text, 
                        current_voice
                    )
                    
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
                      # Edge TTS typically returns MP3 data, but we need WAV format
                    # Convert to WAV format using pydub
                    self.logger.info(f"Converting Edge TTS audio data to WAV format (original size: {len(audio_data)/1024:.1f} KB)")
                    
                    # Offload the pydub conversion to a separate thread
                    wav_data, audio_length_seconds = await self._run_blocking_task(self._do_convert_mp3_to_wav, audio_data)

                    if wav_data == audio_data: # This means conversion failed and original was returned
                        self.logger.warning("Conversion to WAV failed, original MP3 data returned.")
                    else:
                        self.logger.info(f"Successfully converted to WAV format (processed in thread)")

                    self.logger.info(f"Audio length: {audio_length_seconds:.2f} seconds ({len(wav_data)} bytes)")
                    return wav_data
                    
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
                communicate = edge_tts.Communicate("This is a test.", "en-US-GuyNeural") # Using a known good voice
                buffer = io.BytesIO()
                async for chunk_data in communicate.stream(): # Renamed chunk to chunk_data
                    if chunk_data["type"] == "audio":
                        buffer.write(chunk_data["data"])
                buffer.seek(0)
                audio_data = buffer.read()
                
                if len(audio_data) > 0:
                    self.logger.info("Generated fallback audio with minimal text")
                    # Convert this fallback audio to WAV as well
                    wav_data, _ = await self._run_blocking_task(self._do_convert_mp3_to_wav, audio_data)
                    return wav_data
            except Exception as e:
                self.logger.error(f"Even minimal text generation failed: {str(e)}")
            
            # All attempts failed
            raise RuntimeError(f"Failed to generate speech with Edge TTS: {error_message}")

        return await self.run_task_with_keepalive(websocket, _actual_generate())

    def _do_convert_mp3_to_wav(self, mp3_data: bytes) -> tuple[bytes, float]:
        """Synchronous part of converting MP3 data to WAV using pydub."""
        self.logger.info(f"Thread: Converting Edge TTS audio data to WAV format (original size: {len(mp3_data)/1024:.1f} KB)")
        try:
            # Load MP3 data into AudioSegment
            audio_segment = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
            
            # Export as WAV to a bytes buffer
            wav_buffer = io.BytesIO()
            audio_segment.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            wav_data = wav_buffer.read()
            
            # Calculate audio length in seconds
            audio_length_seconds = len(audio_segment) / 1000  # AudioSegment length is in milliseconds
            
            self.logger.info(f"Thread: Successfully converted to WAV format: {len(wav_data)/1024:.1f} KB")
            self.logger.info(f"Thread: Audio length: {audio_length_seconds:.2f} seconds ({len(wav_data)} bytes)")
            return wav_data, audio_length_seconds
            
        except Exception as conv_error:
            self.logger.error(f"Thread: Error converting MP3 to WAV: {str(conv_error)}")
            self.logger.warning("Thread: Returning original MP3 data without conversion due to error.")
            return mp3_data, len(mp3_data) / 48000 # Approximate length for MP3
