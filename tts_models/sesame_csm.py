import io
import logging
import torchaudio
import asyncio # Add asyncio import
from .base_model import BaseTTSModel
from model_loader import ModelLoader

class SesameCSMModel(BaseTTSModel):
    """Sesame CSM-1B model implementation"""
    
    def __init__(self):
        """Initialize the Sesame CSM model"""
        self.logger = logging.getLogger("SesameCSMModel")
        self.ready = False
        self.csm_generator = None
        self.max_audio_length_ms = 150000  # Default to 2.5 minutes
        self.sample_rate = 24000  # Default sample rate
        self.model_loader = ModelLoader(logger=self.logger)
    
    def _do_load_csm_model(self):
        """Synchronous part of loading the CSM model."""
        self.logger.info("Thread: Starting CSM-1B model loading process...")
        csm_gen = self.model_loader.load_csm_model() # This can block (download, disk I/O, CPU)
        self.logger.info("Thread: CSM-1B model loading process finished.")
        return csm_gen

    async def load(self) -> bool:
        """Load the CSM-1B TTS model, offloading blocking parts to a thread."""
        self.logger.info("Preparing to load CSM-1B TTS model...")
        
        self.csm_generator = await self._run_blocking_task(self._do_load_csm_model)
        
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
    
    def is_ready(self) -> bool:
        """Check if the model is ready"""
        return self.ready
    
    def get_sample_rate(self) -> int:
        """Get the sample rate of the generated audio"""
        return self.sample_rate
    
    @property
    def model_name(self) -> str:
        """Get the name of the model"""
        return "Sesame CSM-1B"
    
    @property
    def supported_speakers(self) -> dict:
        """Get the supported speakers"""
        return {
            0: "Male voice",
            1: "Female voice"
        }
    
    async def generate_speech(self, text: str, speaker: int = 0, lang: str = "en-US", **kwargs) -> bytes:
        """Generate speech from text, offloading CPU-bound work to a separate thread."""
        # max_audio_length_ms = kwargs.get("max_audio_length_ms", self.max_audio_length_ms) # Removed parameter
        
        # Check language support
        if lang != "en-US":
            raise ValueError(f"Sesame CSM model only supports 'en-US' language, but received '{lang}'")
            
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        if not self.ready:
            if not await self.load(): # load() is already async and handles its own threading if necessary
                raise RuntimeError("Model failed to load. Check logs for details.")
        
        text_length = len(text)
        self.logger.info(f"Preparing to generate speech for speaker {speaker}: '{text[:100]}...' ({text_length} chars)")

    def _do_generate_and_encode_csm(self, text: str, speaker: int, text_length: int):
        """Synchronous part of generating and encoding speech for CSM."""
        # Use a fixed large value for max_audio_length_ms for stability, as the parameter is removed
        max_audio_length = 180000  # Maximum 3 minutes for stability
        self.logger.info(f"Thread: Using fixed internal max audio length for CSM: {max_audio_length} ms")
        
        # Debug log about actual generation
        self.logger.info(f"Thread: Calling CSM model with text: '{text[:100]}...' ({text_length} chars)")
        self.logger.info(f"Thread: Estimated audio length: ~{(text_length / 20) * 1000:.0f} ms ({text_length / 20:.1f} seconds) at average speech rate")
        
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
            
        self.logger.info(f"Thread: Generated audio tensor with shape: {audio_tensor.shape}")
        if hasattr(audio_tensor, 'shape') and len(audio_tensor.shape) > 0:
            duration_seconds = audio_tensor.shape[-1] / self.sample_rate
            self.logger.info(f"Thread: Estimated audio duration: {duration_seconds:.2f} seconds")
            if duration_seconds < (text_length / 20) * 0.8:  # If much shorter than expected
                self.logger.warning(f"Thread: WARNING: Generated audio seems too short ({duration_seconds:.2f}s) for the input text ({text_length} chars)")
        
        # Convert to WAV format using torchaudio
        wav_io = io.BytesIO()
        
        # Ensure proper format - tensor needs to be 2D [channels, samples]
        audio_tensor_2d = audio_tensor.unsqueeze(0).cpu()
        
        torchaudio.save(wav_io, audio_tensor_2d, self.sample_rate, format='wav')
        wav_io.seek(0)
        return wav_io.read()

    async def generate_speech(self, text: str, speaker: int = 0, lang: str = "en-US", **kwargs) -> bytes:
        """Generate speech from text, offloading CPU-bound work to a separate thread."""
        # max_audio_length_ms = kwargs.get("max_audio_length_ms", self.max_audio_length_ms) # Removed parameter
        
        # Check language support
        if lang != "en-US":
            raise ValueError(f"Sesame CSM model only supports 'en-US' language, but received '{lang}'")
            
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        if not self.ready:
            if not await self.load(): # load() is already async and handles its own threading if necessary
                raise RuntimeError("Model failed to load. Check logs for details.")
        
        text_length = len(text)
        self.logger.info(f"Preparing to generate speech for speaker {speaker}: '{text[:100]}...' ({text_length} chars)")

        try:
            # Verify the CSM generator exists
            if self.csm_generator is None:
                raise RuntimeError("CSM generator model is not loaded")

            # Offload the blocking operations to a separate thread
            self.logger.info("Offloading CSM generation and encoding to a separate thread.")
            wav_bytes = await self._run_blocking_task(self._do_generate_and_encode_csm, text, speaker, text_length)
            
            # Success
            wav_size_kb = len(wav_bytes) / 1024
            self.logger.info(f"Converted audio to WAV format, size: {wav_size_kb:.1f} KB ({len(wav_bytes)} bytes) (processed in thread)")
            return wav_bytes
            
        except Exception as e:
            self.logger.error(f"Error generating TTS audio: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Propagate the error instead of falling back to synthetic audio
            raise RuntimeError(f"Failed to generate speech: {str(e)}")
