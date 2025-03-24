import io
import time
import logging
import torch
import torchaudio
from typing import Tuple

from csm.generator import load_csm_1b
import csm.watermarking
import silentcipher_patch
from model_loader import ModelLoader

class TTSGenerator:
    """
    Text-to-Speech generator using the CSM-1B model.
    """
    
    _generator = None
    _sample_rate = None
    _watermarker = None

    def __init__(self, device=None, max_audio_length_ms=30000):
        """
        Initialize the TTS Generator.
        
        Args:
            device: Device to use for model inference ('cuda' or 'cpu')
            max_audio_length_ms: Maximum audio length in milliseconds
        """
        self.logger = logging.getLogger("TTSGenerator")
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_audio_length_ms = max_audio_length_ms
        
        # Initialize the model loader
        self.model_loader = ModelLoader()
        
        self.logger.info(f"TTS Generator initialized on device: {self.device}")
            
    def load_model(self) -> bool:
        """
        Load the CSM-1B TTS model. Reuse if already loaded.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if TTSGenerator._generator is None:
            self.logger.info(f"Loading CSM-1B model on {self.device}...")
            try:
                # Apply the patch for silentcipher torch.load before model loading
                silentcipher_patch.apply_patch()
                
                # Initialize watermarker only once and store it at class level
                if TTSGenerator._watermarker is None:
                    self.logger.info(f"Initializing watermarker on {self.device}...")
                    TTSGenerator._watermarker = csm.watermarking.load_watermarker(device=self.device)
                
                # Get model path and load model
                model_path = self.model_loader.get_model_path()
                TTSGenerator._generator = load_csm_1b(device=self.device, model_path=model_path)
                TTSGenerator._is_model_loaded = True
                
                # Remove the patch after model loading
                silentcipher_patch.remove_patch()
                
                self.logger.info("Model loaded successfully.")
            except Exception as e:
                # Ensure the patch is removed even if an exception occurs
                silentcipher_patch.remove_patch()
                self.logger.error(f"Error loading model: {e}")
                import traceback
                self.logger.error(f"Stack trace: {traceback.format_exc()}")
                return False
        else:
            self.logger.info("Reusing already loaded model.")
        return True

    def generate_speech(self, text: str, speaker: int = 0, timeout: int = 120) -> Tuple[torch.Tensor, int]:
        """
        Generate speech audio tensor from text with timeout and progress reporting.
        
        Args:
            text: Text to convert to speech
            speaker: Speaker ID (0 for male, 1 for female)
            timeout: Maximum time in seconds to wait for generation
            
        Returns:
            Tuple containing (audio_tensor, sample_rate)
            
        Raises:
            RuntimeError: If the model is not loaded or generation times out
        """
        if TTSGenerator._generator is None:
            self.logger.error("Model not loaded. Call load_model() first.")
            raise RuntimeError("Model not loaded")
        
        # Log text length for diagnostics
        self.logger.info(f"Generating speech for speaker {speaker}: '{text}' ({len(text)} chars)")
        
        # Start time for timeout tracking
        start_time = time.time()
        
        try:
            # Set a watchdog thread if possible
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                # Only use watchdog on CUDA to avoid affecting CPU performance
                prev_handler = None
                if hasattr(torch.cuda, 'set_watchdog_interval'):
                    prev_handler = torch.cuda.get_watchdog_interval()
                    # Check every 5 seconds if we've exceeded timeout
                    torch.cuda.set_watchdog_interval(5000)
            
            # Generate the audio
            audio = TTSGenerator._generator.generate(
                text=text,
                speaker=speaker,
                context=[],
                max_audio_length_ms=self.max_audio_length_ms,
            )
            
            # Check if generation took too long
            elapsed = time.time() - start_time
            if elapsed > timeout:
                self.logger.warning(f"Audio generation completed, but exceeded timeout: {elapsed:.1f}s > {timeout}s")
            else:
                self.logger.info(f"Audio generation completed in {elapsed:.1f}s")
            
            return audio, TTSGenerator._sample_rate
            
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"Error generating speech after {elapsed:.1f}s: {str(e)}")
            raise
        finally:
            # Restore previous watchdog interval if changed
            if hasattr(torch, 'cuda') and torch.cuda.is_available() and 'prev_handler' in locals() and prev_handler is not None:
                if hasattr(torch.cuda, 'set_watchdog_interval'):
                    torch.cuda.set_watchdog_interval(prev_handler)

    def generate_wav_bytes(self, text: str, speaker: int = 0, timeout: int = 120) -> bytes:
        """
        Generate speech audio as WAV bytes with timeout.
        
        Args:
            text: Text to convert to speech
            speaker: Speaker ID (0 for male, 1 for female)
            timeout: Maximum time in seconds to wait for generation
            
        Returns:
            WAV audio data as bytes
            
        Raises:
            RuntimeError: If audio generation fails
        """
        try:
            start_time = time.time()
            audio, sample_rate = self.generate_speech(text, speaker, timeout)
            
            self.logger.info("Converting audio tensor to WAV format...")
            buffer = io.BytesIO()
            torchaudio.save(buffer, audio.unsqueeze(0).cpu(), sample_rate, format="wav")
            buffer.seek(0)
            
            wav_bytes = buffer.read()
            self.logger.info(f"WAV generation complete: {len(wav_bytes)} bytes")
            
            return wav_bytes
        except Exception as e:
            self.logger.error(f"Error in generate_wav_bytes: {str(e)}")
            raise

    @property
    def sample_rate(self) -> int:
        """Get the sample rate of the loaded model."""
        if TTSGenerator._sample_rate is None:
            raise RuntimeError("Model not loaded, sample rate unavailable")
        return TTSGenerator._sample_rate