import io
import pytest
import asyncio
import wave

from tts_models.edge_tts import EdgeTTSModel


@pytest.mark.asyncio
async def test_edge_tts_wav_conversion(logger):
    """Test that Edge TTS model correctly converts MP3 audio to WAV format."""
    # Create an instance of the Edge TTS model
    model = EdgeTTSModel()
    await model.load() # Ensure load is called, even if it's a no-op for readiness
    
    # Check if the model is ready
    assert model.is_ready(), "Edge TTS model should be ready after load"
    
    # Generate speech
    test_text = "This is a test of the Edge TTS MP3 to WAV conversion."
    logger.info(f"Generating speech with text: {test_text}")
    
    # Generate speech using the Edge TTS model
    audio_data = await model.generate_speech(test_text, speaker=0)
    
    logger.info(f"Generated {len(audio_data)/1024:.2f} KB of audio data")
    
    # Verify the audio data is not empty
    assert len(audio_data) > 0, "Generated audio data should not be empty"
    
    # Verify that the audio data is a valid WAV file
    logger.info("Verifying that the generated audio is a valid WAV file")
    with io.BytesIO(audio_data) as audio_io:
        try:
            with wave.open(audio_io, 'rb') as wav_file:
                # Check WAV file properties
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                
                logger.info(f"WAV properties: {channels} channels, {sample_width} bytes/sample, {frame_rate} Hz, {n_frames} frames")
                
                # Verify it's a valid mono 16-bit WAV file
                assert channels == 1, f"Expected mono audio, got {channels} channels"
                assert sample_width == 2, f"Expected 16-bit audio (2 bytes/sample), got {sample_width}"
                assert frame_rate == 24000, f"Expected 24000 Hz sample rate, got {frame_rate}"
                assert n_frames > 0, "WAV file should have at least one frame"
                
                logger.info("Successfully verified WAV file format")
                
        except wave.Error as e:
            pytest.fail(f"Audio data is not a valid WAV file: {str(e)}")
            
    logger.info("Edge TTS MP3 to WAV conversion test passed")


@pytest.mark.asyncio
async def test_edge_tts_load_method(logger):
    """Test the load method of EdgeTTSModel."""
    model = EdgeTTSModel()
    logger.info("Testing EdgeTTSModel load method.")
    loaded = await model.load()
    assert loaded, "Model should report successful loading."
    assert model.is_ready(), "Model should be ready after load."
    logger.info("EdgeTTSModel load method test passed.")


@pytest.mark.asyncio
@pytest.mark.parametrize("lang, speaker_id, test_text_snippet", [
    ("en-US", 1, "Hello world, this is a test."),
    ("ja-JP", 0, "こんにちは、これはテストです。"),
    # ("en-US", 3, "Testing another US voice."), # Example of another speaker if available
])
async def test_edge_tts_generation_languages_speakers(logger, lang, speaker_id, test_text_snippet):
    """Test Edge TTS generation with different languages and speakers."""
    model = EdgeTTSModel()
    await model.load()
    assert model.is_ready(), "Model should be ready."

    logger.info(f"Generating speech with lang='{lang}', speaker_id={speaker_id}, text='{test_text_snippet}'")
    
    try:
        audio_data = await model.generate_speech(test_text_snippet, speaker=speaker_id, lang=lang)
        assert len(audio_data) > 0, "Generated audio data should not be empty."

        # Basic WAV validation
        with io.BytesIO(audio_data) as audio_io:
            with wave.open(audio_io, 'rb') as wav_file:
                assert wav_file.getnchannels() == 1, "Audio should be mono."
                assert wav_file.getsampwidth() == 2, "Audio should be 16-bit."
                assert wav_file.getframerate() == 24000, "Sample rate should be 24000 Hz."
        logger.info(f"Successfully generated and validated audio for lang='{lang}', speaker_id={speaker_id}.")
    except Exception as e:
        pytest.fail(f"Error during speech generation for lang='{lang}', speaker_id={speaker_id}: {str(e)}")


@pytest.mark.asyncio
async def test_edge_tts_supported_languages_and_voices(logger):
    """Test the supported_languages_and_voices property."""
    model = EdgeTTSModel()
    await model.load()
    supported = model.supported_languages_and_voices
    logger.info(f"Supported languages and voices: {supported}")
    
    assert isinstance(supported, dict), "Should return a dictionary."
    assert "en-US" in supported, "en-US should be a supported language."
    assert isinstance(supported["en-US"], dict), "Language entry should be a dictionary of speakers."
    assert 0 in supported["en-US"], "Speaker 0 should be available for en-US."
    assert isinstance(supported["en-US"][0], str), "Speaker description should be a string."
    
    assert "ja-JP" in supported, "ja-JP should be a supported language."
    assert 0 in supported["ja-JP"], "Speaker 0 should be available for ja-JP."
    logger.info("supported_languages_and_voices property test passed.")


@pytest.mark.asyncio
async def test_edge_tts_supported_speakers_default_lang(logger):
    """Test the supported_speakers property (for default language en-US)."""
    model = EdgeTTSModel()
    await model.load()
    speakers = model.supported_speakers
    logger.info(f"Supported speakers for default language (en-US): {speakers}")
    
    assert isinstance(speakers, dict), "Should return a dictionary."
    # Based on EdgeTTSModel.VOICE_MAPPINGS
    expected_en_us_speakers = model.VOICE_MAPPINGS.get("en-US", {})
    assert len(speakers) == len(expected_en_us_speakers), \
        f"Number of speakers for en-US should match VOICE_MAPPINGS: expected {len(expected_en_us_speakers)}, got {len(speakers)}"
    
    for speaker_id in expected_en_us_speakers.keys():
        assert speaker_id in speakers, f"Speaker ID {speaker_id} from VOICE_MAPPINGS should be in supported_speakers."
        assert isinstance(speakers[speaker_id], str), f"Description for speaker {speaker_id} should be a string."
    logger.info("supported_speakers property test passed.")


@pytest.mark.asyncio
@pytest.mark.parametrize("input_lang, expected_output_lang, should_raise_error", [
    ("en-US", "en-US", False),
    ("en_US", "en-US", False),
    ("en", "en-US", False), # Assuming en-US is the first 'en' match
    ("ja-JP", "ja-JP", False),
    ("ja_jp", "ja-JP", False),
    ("ja", "ja-JP", False), # Assuming ja-JP is the first 'ja' match
    ("fr-FR", None, True), # Assuming fr-FR is not in VOICE_MAPPINGS
    ("es", None, True),    # Assuming no 'es' entry in VOICE_MAPPINGS
    ("", "en-US", False), # Empty string defaults to en-US
    (None, "en-US", False), # None defaults to en-US
])
async def test_edge_tts_map_language_code(logger, input_lang, expected_output_lang, should_raise_error):
    """Test the _map_language_code method of EdgeTTSModel."""
    model = EdgeTTSModel()
    # No load needed as _map_language_code is synchronous and doesn't depend on loaded state
    
    logger.info(f"Testing _map_language_code with input: '{input_lang}'")
    
    if should_raise_error:
        with pytest.raises(ValueError) as excinfo:
            model._map_language_code(input_lang)
        logger.info(f"Correctly raised ValueError for '{input_lang}': {excinfo.value}")
    else:
        mapped_lang = model._map_language_code(input_lang)
        assert mapped_lang == expected_output_lang, \
            f"For input '{input_lang}', expected '{expected_output_lang}', but got '{mapped_lang}'"
        logger.info(f"Correctly mapped '{input_lang}' to '{mapped_lang}'")
    logger.info(f"_map_language_code test for '{input_lang}' passed.")
