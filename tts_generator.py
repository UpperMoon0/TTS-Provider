import os
import sys
import io
import logging
import torch
import torchaudio
from dotenv import load_dotenv
from huggingface_hub import HfFolder, snapshot_download

from csm.generator import load_csm_1b  # Make sure this import is present
from csm.watermarking import watermark, load_watermarker

# Load environment variables from .env file
load_dotenv()

class TTSGenerator:
    # Class-level variables to hold the loaded model and its sample rate.
    _generator = None
    _sample_rate = None

    def __init__(self, device=None, max_audio_length_ms=30000):
        self.logger = logging.getLogger("TTSGenerator")
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_audio_length_ms = max_audio_length_ms

        # Simplify to just use HF_HOME directly
        hf_home = os.path.expandvars(os.getenv("HF_HOME", ""))
        if hf_home:
            # Use the standard HF_HOME/hub path
            self.hf_cache_dir = os.path.join(hf_home, "hub")
            os.makedirs(self.hf_cache_dir, exist_ok=True)
            self.logger.info(f"Using HF_HOME/hub: {self.hf_cache_dir}")
        else:
            self.hf_cache_dir = ""
            self.logger.warning("HF_HOME not set. Using default cache location.")

        # Get Hugging Face token from .env
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            HfFolder.save_token(hf_token)
            self.logger.info("Configured Hugging Face token.")

            # Check if model is already cached
            if self.hf_cache_dir:
                model_path = os.path.join(self.hf_cache_dir, "models--sesame--csm-1b")
                if os.path.exists(model_path):
                    self.logger.info(f"Using cached model from: {model_path}")
                else:
                    self.logger.info("Model not found in cache. Downloading...")
                    
                # Pre-download model files if not already cached
                try:
                    snapshot_download(
                        "sesame/csm-1b",
                        cache_dir=self.hf_cache_dir,
                        token=hf_token
                    )
                    self.logger.info("Model files are cached.")
                except Exception as e:
                    self.logger.warning(f"Model caching warning: {e}")

        self.logger.info(f"TTS Generator initialized on device: {self.device}")

    def load_model(self):
        """Load the CSM-1B TTS model. Reuse if already loaded."""
        if TTSGenerator._generator is None:
            self.logger.info(f"Loading CSM-1B model on {self.device}...")
            try:
                # Rely on environment variables already loaded from .env
                # Not setting HF_HOME directly in code
                
                # Log the current HF_HOME for debugging
                hf_home = os.getenv("HF_HOME", "")
                if hf_home:
                    self.logger.info(f"Using HF_HOME from environment: {hf_home}")
                else:
                    self.logger.warning("HF_HOME not set in environment variables")
                
                # Load model with just the device parameter
                TTSGenerator._generator = load_csm_1b(device=self.device)
                TTSGenerator._sample_rate = TTSGenerator._generator.sample_rate
                self.logger.info("Model loaded successfully.")
            except Exception as e:
                self.logger.error(f"Error loading model: {e}")
                return False
        else:
            self.logger.info("Reusing already loaded model.")
        return True

    def generate_speech(self, text, speaker=0):
        """
        Generate speech audio tensor from text.
        Raises:
            RuntimeError if the model is not loaded.
        """
        if TTSGenerator._generator is None:
            self.logger.error("Model not loaded. Call load_model() first.")
            raise RuntimeError("Model not loaded")
        
        self.logger.info(f"Generating speech for speaker {speaker}: '{text}'")
        audio = TTSGenerator._generator.generate(
            text=text,
            speaker=speaker,
            context=[],
            max_audio_length_ms=self.max_audio_length_ms,
        )
        self.logger.info("Audio generation completed.")
        return audio, TTSGenerator._sample_rate

    def generate_wav_bytes(self, text, speaker=0):
        """Generate speech audio as WAV bytes."""
        audio, sample_rate = self.generate_speech(text, speaker)
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio.unsqueeze(0).cpu(), sample_rate, format="wav")
        buffer.seek(0)
        return buffer.read()

    @property
    def sample_rate(self):
        """Get the sample rate of the loaded model."""
        if TTSGenerator._sample_rate is None:
            raise RuntimeError("Model not loaded, sample rate unavailable")
        return TTSGenerator._sample_rate