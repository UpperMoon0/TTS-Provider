import os
import json
import time
import asyncio
import pytest
import tempfile
import websockets

# Test constants
TEST_TEXT = "This is a test of the text-to-speech system from the client integration test."
TEST_SAMPLE_RATE = 24000

@pytest.mark.asyncio
async def test_client_connection(tts_server, logger):
    """Test that the client can connect to the server"""
    server_info = await anext(tts_server)
    port = server_info["port"]
    
    logger.info(f"Testing client connection to server on port {port}")
    
    # Use explicit IPv4 localhost
    connect_host = '127.0.0.1'
    uri = f"ws://{connect_host}:{port}"
    
    # Add retry mechanism for connection
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            # Attempt connection using client connection logic
            async with websockets.connect(
                uri, 
                max_size=10*1024*1024,
                ping_interval=None,
                open_timeout=5
            ) as websocket:
                # Try a simple ping
                pong = await websocket.ping()
                await asyncio.wait_for(pong, timeout=5)
                
                logger.info("Client connection test successful")
                assert True
                return
        except (ConnectionRefusedError, websockets.exceptions.InvalidMessage, 
                asyncio.TimeoutError, OSError) as e:
            logger.warning(f"Connection attempt {attempt+1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Waiting {retry_delay}s before retry...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error("All connection attempts failed")
                raise

@pytest.mark.asyncio
async def test_client_tts_generation(tts_server, logger):
    """Test text-to-speech generation through the client workflow"""
    server_info = await anext(tts_server)
    port = server_info["port"]
    server = server_info["server"]
    
    # Ensure the mock generator has the model_name attribute for speaker mapping
    if not hasattr(server.generator, 'model_name'):
        server.generator.model_name = 'edge'
    
    # Create a temp file for output
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        output_file = temp_file.name
    
    try:
        # Use explicit IPv4 localhost instead of default which might try IPv6
        connect_host = '127.0.0.1'
        uri = f"ws://{connect_host}:{port}"
        logger.info(f"Testing TTS generation via client to {uri}")
        
        # Add retry logic for connecting to the server
        max_retries = 3
        retry_delay = 2
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # Wait a moment to ensure server is fully started
                await asyncio.sleep(1)
                
                async with websockets.connect(
                    uri, 
                    max_size=10*1024*1024,
                    ping_interval=None,
                    open_timeout=10
                ) as websocket:
                    # Send request using client request format
                    request = {
                        "text": TEST_TEXT,
                        "speaker": 0,
                        "sample_rate": TEST_SAMPLE_RATE
                    }
                    
                    logger.info(f"Sending client request: {json.dumps(request)}")
                    send_time = time.time()
                    await websocket.send(json.dumps(request))
                    
                    # Wait for metadata with timeout - longer for real model
                    metadata_str = await asyncio.wait_for(websocket.recv(), timeout=30)
                    metadata = json.loads(metadata_str)
                    logger.info(f"Received metadata: {json.dumps(metadata)}")
                    
                    # Handle model loading status - longer timeout for real model
                    if metadata.get("status") == "loading":
                        logger.info("Model is loading, waiting for completion...")
                        metadata_str = await asyncio.wait_for(websocket.recv(), timeout=120)
                        metadata = json.loads(metadata_str)
                        logger.info(f"Updated metadata: {json.dumps(metadata)}")
                    
                    # Assert metadata is correct
                    assert metadata.get("status") == "success", f"Expected status 'success', got '{metadata.get('status')}'"
                    
                    # Get audio data - longer timeout for real model inference
                    audio_data = await asyncio.wait_for(websocket.recv(), timeout=60)
                    receive_time = time.time()
                    logger.info(f"Received {len(audio_data)} bytes in {receive_time - send_time:.2f}s")
                    
                    # Write audio to file
                    with open(output_file, "wb") as f:
                        f.write(audio_data)
                    
                    # Verify file exists and has content
                    assert os.path.exists(output_file), "Output file wasn't created"
                    assert os.path.getsize(output_file) > 0, "Output file is empty"
                    
                    # Verify file is a valid WAV using wave module
                    import wave
                    with wave.open(output_file, 'rb') as wav_file:
                        assert wav_file.getnchannels() == 1, "Expected mono audio"
                        assert wav_file.getframerate() == TEST_SAMPLE_RATE, f"Expected sample rate {TEST_SAMPLE_RATE}"
                        assert wav_file.getnframes() > 0, "No audio frames in file"
                    
                    # Skip audio playback to avoid dependencies and slow tests
                    logger.info("Audio playback skipped for faster testing")
                    
                    logger.info("Client TTS generation test passed")
                    break  # Success, exit the retry loop
                    
            except (ConnectionRefusedError, websockets.exceptions.InvalidMessage, 
                    asyncio.TimeoutError, OSError) as e:
                last_exception = e
                logger.warning(f"Connection attempt {attempt+1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Waiting {retry_delay}s before retry...")
                    await asyncio.sleep(retry_delay)
        
        if last_exception and attempt == max_retries - 1:
            logger.error("All connection attempts failed")
            raise last_exception

    finally:
        # Clean up temp file
        if os.path.exists(output_file):
            os.unlink(output_file)
            logger.info(f"Cleaned up temporary file {output_file}")

@pytest.mark.asyncio
async def test_client_error_handling(tts_server, logger):
    """Test client error handling for invalid requests"""
    server_info = await anext(tts_server)
    port = server_info["port"]
    server = server_info["server"]
    
    # Ensure the mock generator has the model_name attribute for speaker mapping
    if not hasattr(server.generator, 'model_name'):
        server.generator.model_name = 'edge'
    
    # Set up mock generator to reject requests without text
    # This simulates the error handling we want to test
    original_method = server.generator.generate_speech
    
    async def modified_generate_speech(*args, **kwargs):
        # Check if text is missing or empty
        if 'text' not in kwargs or not kwargs['text']:
            raise ValueError("Missing required field: text")
        return await original_method(*args, **kwargs)
    
    # Replace the method with our version that does validation
    server.generator.generate_speech = modified_generate_speech
    
    # Use explicit IPv4 localhost
    connect_host = '127.0.0.1'
    uri = f"ws://{connect_host}:{port}"
    logger.info(f"Testing client error handling with {uri}")
    
    # Add retry logic
    max_retries = 3
    retry_delay = 2
    
    try:
        for attempt in range(max_retries):
            try:
                # Give server time to be fully ready
                await asyncio.sleep(1)
                
                async with websockets.connect(
                    uri, 
                    max_size=10*1024*1024,
                    ping_interval=None,
                    open_timeout=10
                ) as websocket:
                    # Send an invalid request (missing text)
                    invalid_request = {
                        "speaker": 0,
                        "sample_rate": TEST_SAMPLE_RATE
                    }
                    
                    logger.info(f"Sending invalid request: {json.dumps(invalid_request)}")
                    await websocket.send(json.dumps(invalid_request))
                    
                    # Get error response with timeout
                    response_str = await asyncio.wait_for(websocket.recv(), timeout=10)
                    response = json.loads(response_str)
                    
                    logger.info(f"Received error response: {json.dumps(response)}")
                    
                    # Verify error response
                    assert "error" in response or response.get("status") == "error", "Expected error in response"
                    
                    logger.info("Client error handling test passed")
                    return
            except (ConnectionRefusedError, websockets.exceptions.InvalidMessage, 
                    asyncio.TimeoutError, OSError) as e:
                logger.warning(f"Connection attempt {attempt+1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Waiting {retry_delay}s before retry...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error("All connection attempts failed")
                    raise
    finally:
        # Restore the original method
        server.generator.generate_speech = original_method
