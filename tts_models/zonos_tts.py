import logging
import io
import os
from typing import Dict # Add Dict for type hinting

import torchaudio

from .base_model import BaseTTSModel
# Attempt to import Zonos, with a fallback/warning if not installed
try:
    from zonos.model import Zonos
    from zonos.conditioning import make_cond_dict, supported_language_codes
    from zonos.utils import DEFAULT_DEVICE
except ImportError:
    Zonos = None
    make_cond_dict = None
    supported_language_codes = [] # Provide an empty list as a fallback
    DEFAULT_DEVICE = "cpu"
    logging.getLogger("ZonosTTSModel").warning(
        "Zonos library not found. Please ensure 'zonos' is installed. "
        "See Zonos documentation for installation instructions."
    )

REFERENCE_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "zonos_reference_audio")


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
            # No longer checking for a single DEFAULT_REFERENCE_AUDIO on load,
            # as speaker-specific files will be checked during generation.
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

    @property
    def model_type(self) -> str:
        """Get the type of the model (e.g., "api" or "local")"""
        return "local"

    def get_sample_rate(self) -> int:
        # Zonos natively outputs at 44.1 kHz
        return 44100

    def is_ready(self) -> bool:
        return self.is_model_loaded and self.model is not None

    def _get_cloned_speaker_options(self) -> Dict[int, str]:
        """Helper to generate speaker options based on files in REFERENCE_AUDIO_DIR."""
        cloned_speakers = {}
        if not os.path.exists(REFERENCE_AUDIO_DIR):
            self.logger.warning(f"Reference audio directory not found: {REFERENCE_AUDIO_DIR}")
            return {0: "Default Cloned (No reference file found)"}

        # Check for default_speaker.wav for speaker 0
        default_speaker_path = os.path.join(REFERENCE_AUDIO_DIR, "default_speaker.wav")
        if os.path.exists(default_speaker_path):
            cloned_speakers[0] = "Cloned (default_speaker.wav)"
        else:
            alt_default_path = os.path.join(REFERENCE_AUDIO_DIR, "0.wav")
            if os.path.exists(alt_default_path):
                cloned_speakers[0] = "Cloned (0.wav)"
            else:
                cloned_speakers[0] = "Cloned (No reference for ID 0)"
        
        # Scan for other numbered speaker files
        for i in range(1, 100):  # Scan for up to 100 numbered speakers
            speaker_file_path = os.path.join(REFERENCE_AUDIO_DIR, f"{i}.wav")
            speaker_file_path_alt = os.path.join(REFERENCE_AUDIO_DIR, f"speaker_{i}.wav")
            
            if os.path.exists(speaker_file_path):
                cloned_speakers[i] = f"Cloned ({i}.wav)"
            elif os.path.exists(speaker_file_path_alt):
                 cloned_speakers[i] = f"Cloned (speaker_{i}.wav)"
        
        if not cloned_speakers: # Fallback if directory is empty
            cloned_speakers[0] = "Cloned (No reference files in directory)"
        return cloned_speakers

    @property
    def supported_speakers(self) -> Dict[int, str]:
        """Get the supported speakers for a default language (e.g., en-us)."""
        # Zonos supports cloned speakers for any of its languages.
        # This property will return the speaker options for 'en-us' as a default.
        all_langs_voices = self.supported_languages_and_voices
        return all_langs_voices.get("en-us", self._get_cloned_speaker_options())


    @property
    def supported_languages_and_voices(self) -> Dict[str, Dict[int, str]]:
        """Get all supported languages and the voices/speakers available for each."""
        langs_and_voices = {}
        cloned_speaker_options = self._get_cloned_speaker_options()
        
        if not supported_language_codes: # Fallback if zonos library not fully loaded
            self.logger.warning("Zonos supported_language_codes not available. Reporting limited language support.")
            return {"en-us": cloned_speaker_options} # Default to en-us

        for lang_code in supported_language_codes:
            langs_and_voices[lang_code.lower()] = cloned_speaker_options
            
        if not langs_and_voices: # Should not happen if supported_language_codes has items
             return {"en-us": cloned_speaker_options}
             
        return langs_and_voices

    def _map_language_code(self, lang_code: str) -> str:
        """Maps standard language codes (e.g., en-US) to Zonos format (e.g., en-us)
           and validates against supported codes."""
        mapped_code = lang_code.lower()
        if supported_language_codes and mapped_code not in supported_language_codes:
            self.logger.warning(
                f"Language code '{mapped_code}' (from '{lang_code}') is not in Zonos supported_language_codes. "
                f"Attempting to use it anyway. Available: {supported_language_codes[:10]}..."
            )
            # Optionally, raise an error or fallback to a default language:
            # raise ValueError(f"Unsupported language code for Zonos: {mapped_code}")
        return mapped_code

    async def _generate_speech_async(self, text: str, speaker: int = 0, lang: str = "en-US", **kwargs) -> bytes:
        if not self.is_ready():
            self.logger.error("Zonos model is not loaded. Cannot generate speech.")
            raise RuntimeError("Zonos model not loaded.")

        if Zonos is None or make_cond_dict is None:
            self.logger.error("Zonos library not fully imported. Cannot generate speech.")
            raise RuntimeError("Zonos library not available.")

        try:
            # Determine reference audio path based on speaker ID
            reference_audio_path = None
            possible_filenames = []
            if speaker == 0:
                possible_filenames.extend([
                    os.path.join(REFERENCE_AUDIO_DIR, "default_speaker.wav"),
                    os.path.join(REFERENCE_AUDIO_DIR, "0.wav")
                ])
            else:
                possible_filenames.extend([
                    os.path.join(REFERENCE_AUDIO_DIR, f"{speaker}.wav"),
                    os.path.join(REFERENCE_AUDIO_DIR, f"speaker_{speaker}.wav")
                ])

            for fname in possible_filenames:
                if os.path.exists(fname):
                    reference_audio_path = fname
                    break
            
            if reference_audio_path is None:
                err_msg = f"Reference audio file for speaker ID {speaker} not found in {REFERENCE_AUDIO_DIR}. Searched for: {possible_filenames}"
                self.logger.error(err_msg)
                raise FileNotFoundError(err_msg)

            self.logger.info(f"Loading reference audio from: {reference_audio_path} for speaker ID {speaker}")
            ref_wav, ref_sr = await self._run_blocking_task(torchaudio.load, reference_audio_path)
            
            self.logger.info(f"Creating speaker embedding for speaker ID {speaker}...")
            speaker_embedding = await self._run_blocking_task(self.model.make_speaker_embedding, ref_wav, ref_sr)
            
            mapped_lang = self._map_language_code(lang) # This now includes validation
            self.logger.info(f"Preparing conditioning for lang '{mapped_lang}' (original: '{lang}')...")
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
