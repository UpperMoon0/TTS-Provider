import io
import logging
import torchaudio
from model_loader import ModelLoader

class TTSGenerator:
    """Text-to-speech generator using CSM-1B model"""
    
    def __init__(self, max_audio_length_ms=None):
        """Initialize the TTS generator"""
        self.logger = logging.getLogger("TTSGenerator")
        self.ready = False
        self.csm_generator = None
        self.max_audio_length_ms = max_audio_length_ms or 120000  # Default to 120 seconds (2 minutes) if None
        self.sample_rate = 24000  # Default sample rate
        self.model_loader = ModelLoader(logger=self.logger)
    
    def load_model(self):
        """Load the CSM-1B TTS model"""
        self.logger.info("Loading CSM-1B TTS model...")
        
        # Use the model loader to load the CSM model
        self.csm_generator = self.model_loader.load_csm_model()
        
        if self.csm_generator is not None:
            # Update sample rate from the loaded model
            if hasattr(self.csm_generator, 'sample_rate'):
                self.sample_rate = self.csm_generator.sample_rate
                self.logger.info(f"Using model sample rate: {self.sample_rate}")
            
            self.ready = True
            self.logger.info("CSM-1B TTS model loaded successfully")
            return True
        else:
            self.logger.error("Failed to load CSM-1B model")
            return False
    
    def is_ready(self):
        """Check if the model is ready"""
        return self.ready
    
    async def generate_speech(self, text, speaker=0, sample_rate=None, max_audio_length_ms=None):
        """Generate speech asynchronously"""
        # Just pass through to the synchronous version
        return self.generate_wav_bytes(text, speaker, max_audio_length_ms)
    
    def generate_wav_bytes(self, text, speaker=0, max_audio_length_ms=None):
        """Generate WAV bytes from text using CSM-1B model"""
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        if not self.ready:
            if not self.load_model():
                raise RuntimeError("Model failed to load. Check logs for details.")
        
        text_length = len(text)
        self.logger.info(f"Generating speech for speaker {speaker}: '{text[:100]}...' ({text_length} chars)")
        
        # Generate audio using CSM-1B
        try:
            # Verify the CSM generator exists
            if self.csm_generator is None:
                raise RuntimeError("CSM generator model is not loaded")
                
            # Get max audio length for the model - use the provided value or fall back to default
            # Force a higher value to ensure we get complete audio
            if max_audio_length_ms is None or max_audio_length_ms < 300000:
                max_audio_length = 300000  # Explicitly set to 5 minutes (300,000 ms)
                self.logger.info(f"Using FORCED max audio length: {max_audio_length} ms (overriding provided value: {max_audio_length_ms})")
            else:
                max_audio_length = max_audio_length_ms
                self.logger.info(f"Using provided max audio length: {max_audio_length} ms")
            
            # Debug log about actual generation
            self.logger.info(f"Calling CSM model with text: '{text[:100]}...' ({text_length} chars)")
            self.logger.info(f"Estimated audio length: ~{(text_length / 20) * 1000:.0f} ms ({text_length / 20:.1f} seconds) at average speech rate")
            
            # Generate audio using CSM-1B
            audio_tensor = self.csm_generator.generate(
                text=text,
                speaker=speaker,
                context=[],  # No context for simple generation
                max_audio_length_ms=max_audio_length,
                temperature=0.8,  # Adjust as needed
                topk=50
            )
            
            # Check if we got a valid tensor back
            if audio_tensor is None:
                raise RuntimeError("CSM model returned None instead of audio tensor")
                
            self.logger.info(f"Generated audio tensor with shape: {audio_tensor.shape}")
            if hasattr(audio_tensor, 'shape') and len(audio_tensor.shape) > 0:
                duration_seconds = audio_tensor.shape[-1] / self.sample_rate
                self.logger.info(f"Estimated audio duration: {duration_seconds:.2f} seconds")
                if duration_seconds < (text_length / 20) * 0.8:  # If much shorter than expected
                    self.logger.warning(f"WARNING: Generated audio seems too short ({duration_seconds:.2f}s) for the input text ({text_length} chars)")
            
            # Convert to WAV format using torchaudio
            wav_io = io.BytesIO()
            
            # Ensure proper format - tensor needs to be 2D [channels, samples]
            audio_tensor_2d = audio_tensor.unsqueeze(0).cpu()
            
            torchaudio.save(wav_io, audio_tensor_2d, self.sample_rate, format='wav')
            wav_io.seek(0)
            wav_bytes = wav_io.read()
            
            # Success
            wav_size_kb = len(wav_bytes) / 1024
            self.logger.info(f"Converted audio to WAV format, size: {wav_size_kb:.1f} KB ({len(wav_bytes)} bytes)")
            return wav_bytes
            
        except Exception as e:
            self.logger.error(f"Error generating TTS audio: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Propagate the error instead of falling back to synthetic audio
            raise RuntimeError(f"Failed to generate speech: {str(e)}")