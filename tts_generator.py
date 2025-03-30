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
        self.max_audio_length_ms = max_audio_length_ms or 30000  # Default to 30 seconds if None
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
    
    async def generate_speech(self, text, speaker=0, sample_rate=None):
        """Generate speech asynchronously"""
        # Just pass through to the synchronous version
        return self.generate_wav_bytes(text, speaker)
    
    def generate_wav_bytes(self, text, speaker=0):
        """Generate WAV bytes from text using CSM-1B model"""
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        if not self.ready:
            if not self.load_model():
                raise RuntimeError("Model failed to load. Check logs for details.")
        
        self.logger.info(f"Generating speech for speaker {speaker}: '{text}' ({len(text)} chars)")
        
        # Generate audio using CSM-1B
        try:
            # Verify the CSM generator exists
            if self.csm_generator is None:
                raise RuntimeError("CSM generator model is not loaded")
                
            # Get max audio length for the model
            max_audio_length = self.max_audio_length_ms
            
            # Debug log about actual generation
            self.logger.info(f"Calling CSM model with text: '{text[:50]}...' (truncated)")
            
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
            
            # Convert to WAV format using torchaudio
            wav_io = io.BytesIO()
            
            # Ensure proper format - tensor needs to be 2D [channels, samples]
            audio_tensor_2d = audio_tensor.unsqueeze(0).cpu()
            
            torchaudio.save(wav_io, audio_tensor_2d, self.sample_rate, format='wav')
            wav_io.seek(0)
            
            # Success
            self.logger.info(f"Converted audio to WAV format, size: {wav_io.getbuffer().nbytes} bytes")
            return wav_io.read()
            
        except Exception as e:
            self.logger.error(f"Error generating TTS audio: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Propagate the error instead of falling back to synthetic audio
            raise RuntimeError(f"Failed to generate speech: {str(e)}")