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

    # This map is based on eSpeak NG identifiers, which Zonos uses for phonemization.
    # It aims to map common inputs to the codes Zonos likely expects for its core supported languages.
    # Keys should be normalized (lowercase, hyphenated).
    PREFERRED_ZONOS_LANG_MAP = {
        # English (Defaulting to American English as per Zonos examples)
        "en": "en-us",
        "english": "en-us",
        "en-us": "en-us",
        "en-gb": "en",       # British English (eSpeak has 'en' for British) - Zonos might prefer en-us still
        # Japanese
        "ja": "ja",
        "japanese": "ja",
        "ja-jp": "ja",
        # Chinese (Defaulting to Mandarin)
        "zh": "cmn",
        "chinese": "cmn",
        "cmn": "cmn",        # Mandarin
        "zh-cn": "cmn",
        "yue": "yue",        # Cantonese
        "zh-hk": "yue",
        # French (Defaulting to France)
        "fr": "fr",
        "french": "fr",
        "fr-fr": "fr",
        "fr-ca": "fr", # Canadian French, map to fr, Zonos might handle regionality internally or not
        # German
        "de": "de",
        "german": "de",
        "de-de": "de",
        # Spanish (Defaulting to Spain)
        "es": "es",
        "spanish": "es",
        "es-es": "es",
        "es-mx": "es-419",   # Latin American Spanish
        "es-us": "es-419",   # US Spanish, map to Latin American
        "es-419": "es-419",
        # Korean
        "ko": "ko",
        "korean": "ko",
        "ko-kr": "ko",
        # Russian
        "ru": "ru",
        "russian": "ru",
        "ru-ru": "ru",
        # Portuguese (Defaulting to Portugal)
        "pt": "pt",
        "portuguese": "pt",
        "pt-pt": "pt",
        "pt-br": "pt-br",    # Brazilian Portuguese
        # Italian
        "it": "it",
        "italian": "it",
        "it-it": "it",
        # Arabic
        "ar": "ar",
        "arabic": "ar",
        # Hindi
        "hi": "hi",
        "hindi": "hi",
        "hi-in": "hi",
        # Dutch
        "nl": "nl",
        "dutch": "nl",
        "nl-nl": "nl",
        # Polish
        "pl": "pl",
        "polish": "pl",
        "pl-pl": "pl",
        # Turkish
        "tr": "tr",
        "turkish": "tr",
        "tr-tr": "tr",
        # Vietnamese (Defaulting to Southern Vietnamese)
        "vi": "vi-vn-x-south",
        "vietnamese": "vi-vn-x-south",
        "vi-vn": "vi-vn-x-south",
    }

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
        """
        Maps standard language codes (e.g., en-US, en_GB, en) to a Zonos-supported format.
        Uses a preferred map first, then falls back to checking supported_language_codes.
        Zonos typically expects codes like 'en-us' or 'ja'.
        """
        if not lang_code:
            self.logger.warning("Empty language code provided, defaulting to 'en-us' (from preferred map).")
            return self.PREFERRED_ZONOS_LANG_MAP.get("en", "en-us")

        normalized_input = lang_code.lower().replace('_', '-')

        # 1. Check preferred explicit mappings
        if normalized_input in self.PREFERRED_ZONOS_LANG_MAP:
            mapped_code = self.PREFERRED_ZONOS_LANG_MAP[normalized_input]
            self.logger.info(
                f"Input lang '{lang_code}' (normalized to '{normalized_input}') mapped to preferred Zonos code '{mapped_code}'."
            )
            return mapped_code
        
        # Also check the base language part (e.g., "en" from "en-au") against preferred map
        lang_part_of_input = normalized_input.split('-')[0]
        if lang_part_of_input in self.PREFERRED_ZONOS_LANG_MAP:
            mapped_code = self.PREFERRED_ZONOS_LANG_MAP[lang_part_of_input]
            self.logger.info(
                f"Base lang part '{lang_part_of_input}' of input '{lang_code}' mapped to preferred Zonos code '{mapped_code}'."
            )
            return mapped_code

        # 2. If not in preferred map, use logic with Zonos's supported_language_codes
        self.logger.info(
            f"Input lang '{lang_code}' (normalized: '{normalized_input}') not in preferred map. "
            "Proceeding with Zonos supported_language_codes list."
        )

        if not supported_language_codes:
            self.logger.warning(
                "Zonos supported_language_codes not available and no preferred mapping found for '{normalized_input}'. "
                f"Using normalized input code '{normalized_input}' directly."
            )
            return normalized_input

        # 2a. Direct match in supported_language_codes
        if normalized_input in supported_language_codes:
            self.logger.info(
                f"Normalized input '{normalized_input}' directly found in Zonos supported_language_codes."
            )
            return normalized_input

        # lang_part_of_input was already calculated above
        
        # 2b. Generic input (e.g., "en"), try to find specific variant in supported_language_codes
        # This applies if normalized_input was generic (e.g. "fr") and not in preferred map.
        if lang_part_of_input == normalized_input:
            for slc in supported_language_codes:
                if slc.startswith(lang_part_of_input + "-"):
                    self.logger.info(
                        f"Generic lang '{normalized_input}' (not in preferred map) mapped to first specific "
                        f"Zonos code in supported_language_codes: '{slc}'."
                    )
                    return slc
            # If the generic part itself (e.g. "es") is in supported_language_codes
            if lang_part_of_input in supported_language_codes:
                self.logger.info(
                    f"Generic lang '{lang_part_of_input}' (not in preferred map) directly found in Zonos supported_language_codes."
                )
                return lang_part_of_input
        
        # 2c. Specific input (e.g., "en-au") not in preferred map and not directly in supported_language_codes.
        #     Try to fall back to another variant of the same base language in supported_language_codes.
        for slc in supported_language_codes:
            if slc.startswith(lang_part_of_input + "-"):
                self.logger.warning(
                    f"Lang '{normalized_input}' (not in preferred map or supported_language_codes) "
                    f"falling back to first available variant for '{lang_part_of_input}' in supported_language_codes: '{slc}'."
                )
                return slc
        
        # Fallback: if the base language part itself (e.g. "es" from "es-mx") is in supported_language_codes
        if lang_part_of_input in supported_language_codes:
            self.logger.warning(
                f"Lang '{normalized_input}' not mappable, falling back to base language part '{lang_part_of_input}' "
                "as it is in supported_language_codes."
            )
            return lang_part_of_input

        # 3. No mapping found through any primary logic.
        self.logger.warning(
            f"Lang '{normalized_input}' (from input '{lang_code}') has no preferred mapping and is not found or mappable "
            f"within Zonos supported_language_codes ({supported_language_codes[:10]}...). "
            f"Attempting to use '{normalized_input}' directly with Zonos."
        )
        return normalized_input

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
