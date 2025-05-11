import io
import os
import pytest
import asyncio
import wave
from typing import Dict
from unittest.mock import MagicMock, AsyncMock
import torch # For creating dummy tensors

from tts_models.sesame_csm import SesameCSMModel

# Mark all tests in this module as asyncio
pytestmark = pytest.mark.asyncio

# Helper to create a mock CSM generator
def create_mock_csm_generator(sample_rate=24000, duration_sec=1):
    mock_csm_gen = MagicMock()
    mock_csm_gen.sample_rate = sample_rate
    
    # Create a dummy float tensor
    num_samples = int(sample_rate * duration_sec)
    mock_audio_tensor_float = torch.randn(num_samples)
    
    # Normalize to [-1, 1] to prevent clipping and ensure full range for int16
    if torch.max(torch.abs(mock_audio_tensor_float)) > 0: # Avoid division by zero for silent tensor
        mock_audio_tensor_normalized = mock_audio_tensor_float / torch.max(torch.abs(mock_audio_tensor_float))
    else:
        mock_audio_tensor_normalized = mock_audio_tensor_float

    # Convert to int16
    mock_audio_tensor_int16 = (mock_audio_tensor_normalized * 32767).to(torch.int16)
    
    mock_csm_gen.generate = MagicMock(return_value=mock_audio_tensor_int16)
    return mock_csm_gen

async def test_sesame_csm_load_method(logger, mocker):
    """Test the load method of SesameCSMModel with mocking."""
    model = SesameCSMModel()
    logger.info("Testing SesameCSMModel load method with mocking.")

    mock_csm_gen_instance = create_mock_csm_generator()
    # Patch the internal method that performs the blocking load
    mocker.patch.object(model, '_do_load_csm_model', return_value=mock_csm_gen_instance)

    loaded = await model.load()
    
    assert loaded, "Model should report successful loading."
    assert model.is_ready(), "Model should be ready after load."
    assert model.csm_generator is not None, "CSM generator instance should be initialized."
    assert model.csm_generator == mock_csm_gen_instance
    logger.info("SesameCSMModel load method test passed with mocking.")

@pytest.mark.parametrize("speaker_id, test_text_snippet", [
    (0, "Hello, this is a test of the male voice."),
    (1, "Hello, this is a test of the female voice."),
])
async def test_sesame_csm_generation(logger, mocker, speaker_id, test_text_snippet):
    """Test SesameCSMModel speech generation with different speakers using mocking."""
    model = SesameCSMModel()

    mock_csm_gen_instance = create_mock_csm_generator()
    mocker.patch.object(model, '_do_load_csm_model', return_value=mock_csm_gen_instance)
    
    # Ensure model is loaded before generation
    logger.info(f"Loading mocked model for generation test (speaker {speaker_id})...")
    await model.load() # This will use the mocked _do_load_csm_model
    assert model.is_ready(), "Model must be ready before generation."

    logger.info(f"Generating speech with speaker_id={speaker_id}, text='{test_text_snippet}'")
    # The _do_generate_and_encode_csm method will now use the mocked csm_generator
    
    try:
        audio_data = await model.generate_speech(test_text_snippet, speaker=speaker_id)
        assert len(audio_data) > 0, "Generated audio data should not be empty."

        # Validate that the mocked generate was called
        mock_csm_gen_instance.generate.assert_called_once()

        with io.BytesIO(audio_data) as audio_io:
            with wave.open(audio_io, 'rb') as wav_file:
                assert wav_file.getnchannels() == 1, "Audio should be mono."
                assert wav_file.getsampwidth() == 2, "Audio should be 16-bit."
                # The sample rate should come from the mock generator
                assert wav_file.getframerate() == mock_csm_gen_instance.sample_rate, \
                    f"Sample rate should be {mock_csm_gen_instance.sample_rate} Hz."
        logger.info(f"Successfully generated and validated audio for speaker_id={speaker_id} with mocking.")
    except Exception as e:
        pytest.fail(f"Error during speech generation for speaker_id={speaker_id} with mocking: {str(e)}")


async def test_sesame_csm_generate_empty_text(logger, mocker):
    """Test that generating speech with empty text raises ValueError with mocking."""
    model = SesameCSMModel()
    
    mock_csm_gen_instance = create_mock_csm_generator()
    mocker.patch.object(model, '_do_load_csm_model', return_value=mock_csm_gen_instance)
    
    await model.load() 
    assert model.is_ready(), "Model should be ready."

    logger.info("Testing generation with empty text (mocked).")
    with pytest.raises(ValueError) as excinfo:
        await model.generate_speech("", speaker=0)
    # The error message "Text input cannot be empty" is raised in tts_models/base_model.py
    # but the check in sesame_csm.py is `if not text.strip(): raise ValueError("Text cannot be empty")`
    # Let's ensure the test matches the actual error message from SesameCSMModel or its parent.
    # The actual error message from SesameCSMModel's generate_speech is "Text cannot be empty"
    assert "Text cannot be empty" in str(excinfo.value) 
    logger.info("Correctly raised ValueError for empty text (mocked).")

async def test_sesame_csm_generate_unsupported_language(logger, mocker):
    """Test generation with an unsupported language using mocking."""
    model = SesameCSMModel()

    mock_csm_gen_instance = create_mock_csm_generator()
    mocker.patch.object(model, '_do_load_csm_model', return_value=mock_csm_gen_instance)

    await model.load()
    assert model.is_ready(), "Model should be ready."

    logger.info("Testing generation with a non-English language (mocked).")
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
