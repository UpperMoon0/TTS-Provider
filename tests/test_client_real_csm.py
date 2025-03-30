import os
import pytest
import asyncio
from pathlib import Path
import logging

from tts_client import TTSClient

pytestmark = [
    pytest.mark.integration
]

@pytest.mark.asyncio
async def test_real_tts_generation(real_tts_server, logger):
    """Test text-to-speech generation with the real CSM model."""
    # Get server info from the fixture
    server_info = await anext(real_tts_server)
    
    # Create the client with extended timeout for real model
    client = TTSClient(
        host=server_info["host"],
        port=server_info["port"],
        timeout=900  # Extended timeout for real model operations (15 minutes)
    )
    
    try:
        # Connect to the server
        await client.connect()
        assert client.is_connected(), "Client should be connected to the server"
        
        # Generate speech - using a short text to minimize test time
        test_text = "Hello, this is a test."
        # Save in a dedicated outputs folder so it's easier to find
        output_dir = Path("outputs/real_csm_tests")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "test_real_output.wav"
        
        # Remove output file if it exists
        if output_path.exists():
            output_path.unlink()
            
        # Generate speech using file mode instead of stream mode
        logger.info("Requesting TTS generation in file mode...")
        await client.generate_speech(
            text=test_text, 
            output_path=str(output_path),
            response_mode="file"  # Use file mode instead of default stream mode
        )
        
        # Verify the output file was created
        assert output_path.exists(), "Output file should have been created"
        assert output_path.stat().st_size > 0, "Output file should not be empty"
        logger.info(f"Real TTS generation successful, audio saved to {output_path}")
        logger.info(f"File size: {output_path.stat().st_size} bytes")
        
        # Keep the file for inspection (don't delete in finally block)
        return str(output_path)
        
    finally:
        # Clean up - only disconnect, don't remove the output file
        if client.is_connected():
            await client.disconnect()
