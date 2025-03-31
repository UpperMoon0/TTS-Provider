import os
import sys
import logging
import asyncio
import socket
import pytest
import io
import numpy as np
import soundfile as sf
import websockets
from unittest.mock import MagicMock
from contextlib import closing
from typing import AsyncGenerator, Generator, Tuple

from tts_server import TTSServer
from tts_generator import TTSGenerator

# Register custom markers
def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark test as integration test")

# Add the command line option for preloading model
def pytest_addoption(parser):
    parser.addoption("--preload-model", action="store_true", default=False,
                     help="Preload the TTS model before running tests")

@pytest.fixture
def logger():
    """Fixture to provide a logger for tests."""
    logger = logging.getLogger("TTS-Test")
    logger.setLevel(logging.WARNING)
    return logger

def find_free_port():
    """Find a free port on localhost."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('localhost', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

@pytest.fixture
def available_port():
    """Fixture to provide an available port for testing."""
    return find_free_port()

@pytest.fixture
def mock_tts_generator():
    """Fixture to provide a mock TTSGenerator."""
    mock_generator = MagicMock(spec=TTSGenerator)
    
    # Set up mock behavior
    mock_generator.is_ready.return_value = True
    mock_generator.load_model.return_value = True
    
    # Create a dummy WAV file with a sine wave for testing
    sample_rate = 24000
    duration = 1  # 1 second of audio
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    wav_io = io.BytesIO()
    sf.write(wav_io, audio, sample_rate, format='WAV')
    wav_io.seek(0)
    wav_bytes = wav_io.read()
    
    # Set up the generate_wav_bytes mock to return the dummy audio
    mock_generator.generate_wav_bytes.return_value = wav_bytes
    
    # Make the async generate_speech method also return the dummy audio
    async def mock_generate_speech(*args, **kwargs):
        return wav_bytes
    
    mock_generator.generate_speech = mock_generate_speech
    mock_generator.sample_rate = sample_rate
    
    return mock_generator

@pytest.fixture
async def tts_server(available_port, mock_tts_generator, logger):
    """Fixture to provide a TTS server with mock generator for unit tests."""
    # Initialize the server
    server = TTSServer(host='localhost', port=available_port)
    
    # Replace the real generator with our mock
    server.generator = mock_tts_generator
    
    # Start the server in a background task
    server_task = None
    server_instance = None
    
    try:
        # We'll use an event to signal when the server is ready
        startup_event = asyncio.Event()
        
        async def run_server():
            nonlocal server_instance
            try:
                async def handler_adapter(websocket):
                    await server.handle_client(websocket, "/")
                    
                server_instance = await websockets.serve(
                    handler_adapter,
                    'localhost',
                    available_port,
                    ping_interval=None,
                    ping_timeout=None,
                    max_size=None,
                    max_queue=None
                )
                
                startup_event.set()
                await asyncio.Future()
            except Exception as e:
                logger.error(f"Server error: {str(e)}")
                startup_event.set()
            
            return server_instance
            
        server_task = asyncio.create_task(run_server())
        await asyncio.wait_for(startup_event.wait(), timeout=5.0)
        
        yield {
            "server": server,
            "port": available_port,
            "host": 'localhost'
        }
        
    finally:
        logger.info("Stopping test server...")
        
        if server_task and not server_task.done():
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error during server shutdown: {e}")
        
        if server_instance:
            server_instance.close()
            await server_instance.wait_closed()

@pytest.fixture
async def real_tts_server(available_port, logger, request):
    """Fixture to provide a TTS server with real generator for integration tests."""
    # Initialize the server with real generator
    server = TTSServer(host='localhost', port=available_port)

    # Start the server in a background task
    server_task = None
    server_instance = None

    try:
        # We'll use an event to signal when the server is ready
        startup_event = asyncio.Event()
        model_loaded = False

        # Optionally preload the model to avoid timeouts during tests
        if request.config.getoption("--preload-model", False):
            logger.info("Preloading TTS model (this may take a while)...")
            # Call the preload_model method directly
            await server.preload_model()
            model_loaded = server.model_loaded
            logger.info(f"TTS model preloaded successfully: {model_loaded}")

        async def run_server():
            nonlocal server_instance
            try:
                async def handler_adapter(websocket):
                    await server.handle_client(websocket, "/")
                    
                server_instance = await websockets.serve(
                    handler_adapter,
                    'localhost',
                    available_port,
                    ping_interval=None,
                    ping_timeout=None,
                    max_size=None,
                    max_queue=None
                )
                
                # Start the queue processor if model is loaded
                if server.model_loaded and server.queue_processor_task is None:
                    server.queue_processor_task = asyncio.create_task(server.process_queued_requests())
                
                startup_event.set()
                await asyncio.Future()
            except Exception as e:
                logger.error(f"Server error: {str(e)}")
                startup_event.set()
            
            return server_instance
            
        server_task = asyncio.create_task(run_server())
        await asyncio.wait_for(startup_event.wait(), timeout=5.0)
        
        yield {
            "server": server,
            "port": available_port,
            "host": 'localhost',
            "model_loaded": model_loaded
        }
        
    finally:
        logger.info("Stopping real TTS server...")
        
        if server_task and not server_task.done():
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error during real server shutdown: {e}")
        
        if server_instance:
            server_instance.close()
            await server_instance.wait_closed()

@pytest.fixture(scope="session", autouse=True)
async def cleanup_after_tests():
    """Cleanup fixture that runs after all tests."""
    yield
