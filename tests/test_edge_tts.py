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
    
    # Check if the model is ready
    assert model.is_ready(), "Edge TTS model should be ready immediately"
    
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
