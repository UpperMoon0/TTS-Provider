import asyncio
import logging
import io
import os

import torch
import torchaudio

from .base_model import BaseTTSModel
# Attempt to import Zonos, with a fallback/warning if not installed
try:
    from zonos.model import Zonos
    from zonos.conditioning import make_cond_dict
    from zonos.utils import DEFAULT_DEVICE
except ImportError:
    Zonos = None
    make_cond_dict = None
    DEFAULT_DEVICE = "cpu"
    logging.getLogger("ZonosTTSModel").warning(
        "Zonos library not found. Please ensure 'zonos' is installed. "
        "See Zonos documentation for installation instructions."
    )

# Define a path for reference audio files. Users will need to place files here.
# For example, for speaker 0, a file named "0.wav" or "default.wav" would be expected.
# This path is relative to the tts_models directory.
REFERENCE_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "zonos_reference_audio")
DEFAULT_REFERENCE_AUDIO = os.path.join(REFERENCE_AUDIO_DIR, "default_speaker.wav")


class ZonosTTSModel(BaseTTSModel):
    """TTS Model for Zonos"""

    def __init__(self):
        super().__init__()
        self.model = None
        self.is_model_loaded = False
        # Ensure the reference audio directory exists
        if not os.path.exists(REFERENCE_AUDIO_DIR):
            try:
                os.makedirs(REFERENCE_AUDIO_DIR)
                self.logger.info(f"Created reference audio directory: {REFERENCE_AUDIO_DIR}")
            except OSError as e:
                self.logger.error(f"Failed to create reference audio directory {REFERENCE_AUDIO_DIR}: {e}")
        
        if Zonos is None:
            self.logger.error("Zonos library is not installed. ZonosTTSModel will not function.")


    async def _load_model_async(self):
        if Zonos is None:
            self.logger.error("Cannot load Zonos model: Zonos library not installed.")
            return False
        try:
            self.logger.info(f"Loading Zonos model (Zyphra/Zonos-v0.1-transformer) onto device: {DEFAULT_DEVICE}...")
            self.model = await self._run_blocking_task(Zonos.from_pretrained, "Zyphra/Zonos-v0.1-transformer", device=DEFAULT_DEVICE)
            self.is_model_loaded = True
            self.logger.info("Zonos model loaded successfully.")
            
            # Check for default reference audio
            if not os.path.exists(DEFAULT_REFERENCE_AUDIO):
                self.logger.warning(
                    f"Default reference audio not found: {DEFAULT_REFERENCE_AUDIO}. "
                    "Voice cloning for speaker 0 might fail or use a fallback if any. "
                    "Please place a WAV file at this location for default speaker."
                )
            return True
        except Exception as e:
            self.logger.error(f"Failed to load Zonos model: {e}", exc_info=True)
            self.is_model_loaded = False
            return False

    async def load(self, websocket=None) -> bool:
        if self.is_model_loaded:
            return True
        
        # Use run_task_with_keepalive if websocket is provided
        if websocket:
            return await self.run_task_with_keepalive(websocket, self._load_model_async())
        else:
            return await self._load_model_async()

    @property
    def model_name(self) -> str:
        return "zonos"

    def get_sample_rate(self) -> int:
        # Zonos natively outputs at 44.1 kHz
        return 44100

    def is_ready(self) -> bool:
        return self.is_model_loaded and self.model is not None

    @property
    def supported_speakers(self) -> dict:
        # Zonos uses voice cloning. We define a single "default" speaker ID.
        # Users would need to place a reference audio file (e.g., default_speaker.wav)
        # in the REFERENCE_AUDIO_DIR.
        return {
            0: "Default (Voice clone from reference audio)"
            # Potentially, more IDs could map to different files in REFERENCE_AUDIO_DIR
        }

    def _map_language_code(self, lang_code: str) -> str:
        """Maps standard language codes (e.g., en-US) to Zonos format (e.g., en-us)."""
        return lang_code.lower()

    async def _generate_speech_async(self, text: str, speaker: int = 0, lang: str = "en-US", **kwargs) -> bytes:
        if not self.is_ready():
            self.logger.error("Zonos model is not loaded. Cannot generate speech.")
            raise RuntimeError("Zonos model not loaded.")

        if Zonos is None or make_cond_dict is None:
            self.logger.error("Zonos library not fully imported. Cannot generate speech.")
            raise RuntimeError("Zonos library not available.")

        try:
            # Determine reference audio path based on speaker ID
            # For now, only speaker 0 is supported and maps to DEFAULT_REFERENCE_AUDIO
            reference_audio_path = DEFAULT_REFERENCE_AUDIO
            if speaker != 0:
                self.logger.warning(f"Speaker ID {speaker} requested, but only speaker 0 (default) is currently configured for Zonos. Using default reference.")
            
            if not os.path.exists(reference_audio_path):
                self.logger.error(f"Reference audio file not found: {reference_audio_path}. Cannot perform voice cloning.")
                raise FileNotFoundError(f"Required reference audio not found: {reference_audio_path}")

            self.logger.info(f"Loading reference audio from: {reference_audio_path}")
            ref_wav, ref_sr = await self._run_blocking_task(torchaudio.load, reference_audio_path)
            
            self.logger.info("Creating speaker embedding...")
            speaker_embedding = await self._run_blocking_task(self.model.make_speaker_embedding, ref_wav, ref_sr)
            
            mapped_lang = self._map_language_code(lang)
            self.logger.info(f"Preparing conditioning for lang '{mapped_lang}'...")
            cond_dict = make_cond_dict(text=text, speaker=speaker_embedding, language=mapped_lang)
            conditioning = await self._run_blocking_task(self.model.prepare_conditioning, cond_dict)
            
            self.logger.info("Generating audio codes...")
            # Assuming model.generate is synchronous based on example
            codes = await self._run_blocking_task(self.model.generate, conditioning)
            
            self.logger.info("Decoding audio...")
            # Assuming model.autoencoder.decode is synchronous
            audio_tensor = await self._run_blocking_task(lambda c: self.model.autoencoder.decode(c).cpu(), codes)
            
            # Zonos returns stereo, but typically TTS is mono. Taking the first channel.
            # If it's already mono, unsqueeze(0) will handle it for torchaudio.save.
            # The output from decode is typically (batch, channels, samples) or (batch, samples)
            # We take the first item from batch: audio_tensor[0]
            # If it's stereo (e.g., shape [channels, samples]), take one channel: audio_tensor[0][0] or average
            # For simplicity, let's assume it's [samples] or [1, samples] after [0] index
            processed_audio_tensor = audio_tensor[0]
            if processed_audio_tensor.ndim > 1 and processed_audio_tensor.shape[0] > 1: # if it's multi-channel
                 self.logger.info(f"Decoded audio has multiple channels ({processed_audio_tensor.shape[0]}), taking the first one.")
                 processed_audio_tensor = processed_audio_tensor[0, :] # Take first channel
            
            # Ensure it's 2D (batch, samples) for torchaudio.save
            if processed_audio_tensor.ndim == 1:
                processed_audio_tensor = processed_audio_tensor.unsqueeze(0)

            buffer = io.BytesIO()
            await self._run_blocking_task(torchaudio.save, buffer, processed_audio_tensor, self.get_sample_rate(), format="wav")
            
            self.logger.info("Speech generated successfully with Zonos.")
            return buffer.getvalue()

        except FileNotFoundError as fnf_err:
            self.logger.error(f"File not found during Zonos speech generation: {fnf_err}")
            raise  # Re-raise to be caught by server
        except Exception as e:
            self.logger.error(f"Error during Zonos speech generation: {e}", exc_info=True)
            # Consider raising a more specific error or returning None/empty bytes
            raise RuntimeError(f"Zonos speech generation failed: {e}")


    async def generate_speech(self, text: str, speaker: int = 0, lang: str = "en-US", websocket=None, **kwargs) -> bytes:
        if websocket:
            return await self.run_task_with_keepalive(websocket, self._generate_speech_async(text, speaker, lang, **kwargs))
        else:
            return await self._generate_speech_async(text, speaker, lang, **kwargs)
