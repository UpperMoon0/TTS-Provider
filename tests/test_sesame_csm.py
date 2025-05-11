import io
import os
import pytest
import asyncio
import wave
from typing import Dict

from tts_models.sesame_csm import SesameCSMModel

# Mark all tests in this module as asyncio
pytestmark = pytest.mark.asyncio

async def test_sesame_csm_load_method(logger):
    """Test the load method of SesameCSMModel."""
    model = SesameCSMModel()
    logger.info("Testing SesameCSMModel load method.")

    # This might take some time depending on whether the model is cached
    loaded = await model.load()
    
    assert loaded, "Model should report successful loading."
    assert model.is_ready(), "Model should be ready after load."
    assert model.model is not None, "CSM model instance should be initialized."
    logger.info("SesameCSMModel load method test passed.")

@pytest.mark.parametrize("speaker_id, test_text_snippet", [
    (0, "Hello, this is a test of the male voice."),
    (1, "Hello, this is a test of the female voice."),
])
async def test_sesame_csm_generation(logger, speaker_id, test_text_snippet):
    """Test SesameCSMModel speech generation with different speakers."""
    model = SesameCSMModel()

    # Ensure model is loaded before generation
    logger.info(f"Loading model for generation test (speaker {speaker_id})...")
    await model.load()
    assert model.is_ready(), "Model must be ready before generation."

    logger.info(f"Generating speech with speaker_id={speaker_id}, text='{test_text_snippet}'")
    
    try:
        audio_data = await model.generate_speech(test_text_snippet, speaker=speaker_id)
        assert len(audio_data) > 0, "Generated audio data should not be empty."

        with io.BytesIO(audio_data) as audio_io:
            with wave.open(audio_io, 'rb') as wav_file:
                assert wav_file.getnchannels() == 1, "Audio should be mono."
                assert wav_file.getsampwidth() == 2, "Audio should be 16-bit."
                assert wav_file.getframerate() == model.get_sample_rate(), \
                    f"Sample rate should be {model.get_sample_rate()} Hz."
        logger.info(f"Successfully generated and validated audio for speaker_id={speaker_id}.")
    except Exception as e:
        pytest.fail(f"Error during speech generation for speaker_id={speaker_id}: {str(e)}")


async def test_sesame_csm_generate_empty_text(logger):
    """Test that generating speech with empty text raises ValueError."""
    model = SesameCSMModel()
    await model.load() # Load is needed to pass the self.ready check in generate_speech
    assert model.is_ready(), "Model should be ready."

    logger.info("Testing generation with empty text.")
    with pytest.raises(ValueError) as excinfo:
        await model.generate_speech("", speaker=0)
    assert "Text input cannot be empty" in str(excinfo.value)
    logger.info("Correctly raised ValueError for empty text.")

async def test_sesame_csm_generate_unsupported_language(logger):
    """Test that generating speech with an unsupported language (though CSM is primarily English)
       doesn't crash and potentially falls back or handles it gracefully."""
    model = SesameCSMModel()
    await model.load()
    assert model.is_ready(), "Model should be ready."

    logger.info("Testing generation with a non-English language (expecting fallback or graceful handling).")
    # Sesame CSM is primarily for English. How it handles other languages might depend on its internal tokenizer.
    # This test assumes it won't crash but might produce suboptimal or English-interpreted audio.
    # For now, we just check that it doesn't raise an unhandled exception.
    try:
        audio_data = await model.generate_speech("こんにちは世界", speaker=0, lang="ja") # Japanese text
        assert len(audio_data) > 0, "Audio data should be generated even for non-primary language."
        logger.info("Generated audio for non-English text without crashing.")
    except Exception as e:
        pytest.fail(f"Generation with non-English text failed unexpectedly: {str(e)}")


async def test_sesame_csm_supported_languages_and_voices(logger):
    """Test the supported_languages_and_voices property for SesameCSMModel."""
    model = SesameCSMModel()
    # No load needed to check this property as it's static for CSM
    
    supported = model.supported_languages_and_voices
    logger.info(f"Supported languages and voices for SesameCSM: {supported}")
    
    assert isinstance(supported, dict), "Should return a dictionary."
    assert "en-US" in supported, "en-US should be a supported language."
    assert isinstance(supported["en-US"], dict), "Entry for en-US should be a dict of speakers."
    assert 0 in supported["en-US"], "Speaker 0 (male) should be available for en-US."
    assert 1 in supported["en-US"], "Speaker 1 (female) should be available for en-US."
    assert supported["en-US"][0] == "Male voice"  # Updated expected value
    assert supported["en-US"][1] == "Female voice"  # Updated expected value
    logger.info("supported_languages_and_voices property test passed for SesameCSM.")

async def test_sesame_csm_supported_speakers(logger):
    """Test the supported_speakers property for SesameCSMModel."""
    model = SesameCSMModel()
    # No load needed
    
    speakers = model.supported_speakers # This defaults to en-US speakers
    logger.info(f"Supported speakers for SesameCSM (default lang en-US): {speakers}")
    
    assert isinstance(speakers, dict), "Should return a dictionary."
    assert 0 in speakers and speakers[0] == "Male voice"  # Updated expected value
    assert 1 in speakers and speakers[1] == "Female voice"  # Updated expected value
    logger.info("supported_speakers property test passed for SesameCSM.")

@pytest.mark.parametrize("input_lang, expected_csm_code, should_raise_error", [
    ("en-US", "en-US", False),
    ("en_GB", "en-US", False), # Falls back to en-US
    ("en", "en-US", False),    # Falls back to en-US
    ("english", "en-US", False),
    ("ja-JP", "en-US", False), # Non-English falls back to en-US (as CSM is English-focused)
    ("fr", "en-US", False),    # Non-English falls back
    ("", "en-US", False),      # Empty defaults to en-US
    (None, "en-US", False),   # None defaults to en-US
    ("esperanto", "en-US", False), # Unknown language, should default to en-US
])
async def test_sesame_csm_map_language_code(logger, input_lang, expected_csm_code, should_raise_error):
    """Test the _map_language_code method of SesameCSMModel."""
    model = SesameCSMModel()
    # No load needed
    
    logger.info(f"Testing _map_language_code for SesameCSM with input: '{input_lang}'")
    
    if should_raise_error: # SesameCSM's map_language_code currently doesn't raise errors, it defaults.
        with pytest.raises(ValueError) as excinfo:
            model._map_language_code(input_lang)
        assert "could not be mapped" in str(excinfo.value).lower() # Or similar error
        logger.info(f"Correctly raised ValueError for '{input_lang}'.")
    else:
        mapped_lang = model._map_language_code(input_lang)
        assert mapped_lang == expected_csm_code, \
            f"For input '{input_lang}', expected CSM code '{expected_csm_code}', but got '{mapped_lang}'"
        logger.info(f"Correctly mapped '{input_lang}' to '{mapped_lang}' for SesameCSM.")
    logger.info(f"_map_language_code test for SesameCSM with '{input_lang}' passed.")
