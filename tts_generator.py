import os
import sys
import torch
import torchaudio
import io
import logging
import numpy as np
from pathlib import Path

# Add CSM directory to Python path
csm_path = os.path.join(os.path.dirname(__file__), 'csm')
sys.path.append(csm_path)

class TTSGenerator:
    """
    Class to generate speech from text using the CSM-1B model.
    Encapsulates the model loading and speech generation functionality.
    """
    
    def __init__(self, device=None, hf_home=None, hf_token=None, max_audio_length_ms=30000):
        """
        Initialize the TTS Generator
        
        Args:
            device: Device to run the model on (cuda or cpu)
            hf_home: Path to store Hugging Face models
            hf_token: Hugging Face token for accessing gated models
            max_audio_length_ms: Maximum length of audio to generate in milliseconds
        """
        self.logger = logging.getLogger("TTSGenerator")
        
        # Configure Hugging Face environment
        if hf_home:
            os.environ["HF_HOME"] = hf_home
        if hf_token:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
            self.logger.info("Hugging Face token configured for authentication")
        
        # Set device (cuda if available, otherwise cpu)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_audio_length_ms = max_audio_length_ms
        self.generator = None
        self.sample_rate = None
        
        self.logger.info(f"TTS Generator initialized with device: {self.device}")
    
    def load_model(self):
        """Load the CSM-1B TTS model"""
        self.logger.info(f"Loading CSM-1B model on {self.device}...")
        
        # Import the generator module
        from generator import load_csm_1b
        
        # Load the model
        self.generator = load_csm_1b(device=self.device)
        self.sample_rate = self.generator.sample_rate
        self.logger.info("Model loaded successfully")
        
        return self.generator is not None
    
    def generate_speech(self, text, speaker=0):
        """
        Generate speech audio from text
        
        Args:
            text: Text to convert to speech
            speaker: Speaker ID (0 for male, 1 for female)
            
        Returns:
            audio_tensor: Audio data as a tensor
            sample_rate: Sample rate of the audio
        """
        if not self.generator:
            self.logger.error("Model not loaded. Call load_model() first.")
            raise RuntimeError("Model not loaded")
        
        self.logger.info(f"Generating speech for speaker {speaker}")
        self.logger.info(f"Text: '{text}'")
        
        # Generate audio
        audio = self.generator.generate(
            text=text,
            speaker=speaker,
            context=[],
            max_audio_length_ms=self.max_audio_length_ms,
        )
        
        self.logger.info("Audio generation completed")
        return audio, self.sample_rate
    
    def generate_wav_bytes(self, text, speaker=0):
        """
        Generate speech and return as WAV bytes
        
        Args:
            text: Text to convert to speech
            speaker: Speaker ID (0 for male, 1 for female)
            
        Returns:
            wav_bytes: Audio data as WAV bytes
        """
        audio, sample_rate = self.generate_speech(text, speaker)
        
        # Convert to WAV bytes
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio.unsqueeze(0).cpu(), sample_rate, format="wav")
        buffer.seek(0)
        
        return buffer.read()