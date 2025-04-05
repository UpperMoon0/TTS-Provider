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
# Import the cleanup function and our new logging utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from asyncio_helper import cleanup_pending_tasks
from logging_utils import configure_safe_logging, suppress_logging_errors

# Configure safe logging for the entire test session
configure_safe_logging()

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
    
    # Add model_name attribute needed for speaker mapping
    mock_generator.model_name = "edge"
    
    # Create a dummy WAV file with a sine wave for testing
    sample_rate = 24000
    duration = 1  # 1 second of audio
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    wav_io = io.BytesIO()
    sf.write(wav_io, audio, sample_rate, format='WAV')
    wav_io.seek(0)
    wav_bytes = wav_io.read()
    
    # Set up the async generate_speech method to return the dummy audio
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
    queue_processor_task = None
    
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
                
                # Instead of waiting on an infinite Future, use a done_event
                done_event = asyncio.Event()
                try:
                    await done_event.wait()
                except asyncio.CancelledError:
                    pass
                    
            except Exception as e:
                logger.error(f"Server error: {str(e)}")
                startup_event.set()
        
        # Create the server task
        server_task = asyncio.create_task(run_server())
        await asyncio.wait_for(startup_event.wait(), timeout=5.0)
        
        # Start the queue processor if model is ready
        if server.model_loaded and server.queue_processor_task is None:
            server.queue_processor_task = asyncio.create_task(server.process_queued_requests())
            queue_processor_task = server.queue_processor_task
        
        yield {
            "server": server,
            "port": available_port,
            "host": 'localhost'
        }
        
    finally:
        logger.info("Stopping test server...")
        
        # Cancel and clean up the queue processor task if it exists
        if queue_processor_task and not queue_processor_task.done():
            queue_processor_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(queue_processor_task), timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception as e:
                logger.error(f"Error cancelling queue processor: {e}")
        
        # Properly clean up server task
        if server_task and not server_task.done():
            server_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(server_task), timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception as e:
                logger.error(f"Error during server shutdown: {e}")
        
        # Close the server properly
        if server_instance:
            server_instance.close()
            try:
                await asyncio.wait_for(server_instance.wait_closed(), timeout=1.0)
            except asyncio.TimeoutError:
                logger.warning("Server close timed out, continuing cleanup")
            except Exception as e:
                logger.error(f"Error waiting for server to close: {e}")
        
        # Clean up any remaining tasks that might be related to this server
        tasks = [t for t in asyncio.all_tasks() if t != asyncio.current_task() and 
                "handler_adapter" in str(t) or "handle_client" in str(t)]
        for task in tasks:
            task.cancel()
        
        # Allow a short time for task cancellation to complete
        if tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

@pytest.fixture
async def real_tts_server(available_port, logger, request):
    """Fixture to provide a TTS server with real generator for integration tests."""
    # Initialize the server with real generator
    server = TTSServer(host='localhost', port=available_port)
    
    # Ensure generator is initialized 
    if server.generator is None:
        from tts_generator import TTSGenerator
        default_model = os.environ.get("TTS_MODEL", "edge")  # Use edge for faster tests
        server.generator = TTSGenerator(model_name=default_model)
        logger.info(f"Initialized TTS generator with model: {default_model}")

    # Start the server in a background task
    server_task = None
    server_instance = None
    queue_processor_task = None

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
        else:
            # For tests, always make sure the model is ready to avoid connection issues
            logger.info("Initializing TTS model for tests...")
            # Set model_loaded to True since we're using edge TTS which is always ready
            model_loaded = True
            server.model_loaded = True

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
                    nonlocal queue_processor_task
                    queue_processor_task = server.queue_processor_task
                
                startup_event.set()
                
                # Instead of waiting on an infinite Future, use a done_event
                done_event = asyncio.Event()
                try:
                    await done_event.wait()
                except asyncio.CancelledError:
                    pass
                
            except Exception as e:
                logger.error(f"Server error: {str(e)}")
                startup_event.set()
        
        # Create the server task
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
        
        # Cancel and clean up the queue processor task if it exists
        if queue_processor_task and not queue_processor_task.done():
            queue_processor_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(queue_processor_task), timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception as e:
                logger.error(f"Error cancelling queue processor: {e}")
        
        # Properly clean up server task
        if server_task and not server_task.done():
            server_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(server_task), timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception as e:
                logger.error(f"Error during server shutdown: {e}")
        
        # Close the server properly
        if server_instance:
            server_instance.close()
            try:
                await asyncio.wait_for(server_instance.wait_closed(), timeout=1.0)
            except asyncio.TimeoutError:
                logger.warning("Server close timed out, continuing cleanup")
            except Exception as e:
                logger.error(f"Error waiting for server to close: {e}")
        
        # Clean up any remaining tasks that might be related to this server
        tasks = [t for t in asyncio.all_tasks() if t != asyncio.current_task() and 
                "handler_adapter" in str(t) or "handle_client" in str(t)]
        for task in tasks:
            task.cancel()
        
        # Allow a short time for task cancellation to complete
        if tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

# Run this fixture after each test to clean up any pending asyncio tasks
@pytest.fixture(autouse=True, scope="function")
async def cleanup_after_test():
    """Clean up asyncio tasks after each test."""
    yield
    with suppress_logging_errors():
        await cleanup_pending_tasks()

# Configure asyncio to use a specific event loop policy
@pytest.fixture(scope="session", autouse=True)
async def configure_event_loop():
    """Configure the event loop policy for the test session."""
    # Set a longer event loop exception handler that logs but doesn't warn about pending tasks
    def custom_exception_handler(loop, context):
        exception = context.get('exception')
        if exception is not None:
            logging.error(f"Unhandled exception: {exception}")
    
    # Get the current event loop
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(custom_exception_handler)
    
    # Run the tests with this loop
    yield
    
    # Clean up any pending tasks on test completion
    with suppress_logging_errors():
        pending = asyncio.all_tasks(loop)
        current_task = asyncio.current_task(loop)
        
        if pending:
            pending = [task for task in pending if task is not current_task]
            if pending:
                logging.info(f"Cancelling {len(pending)} pending tasks")
                for task in pending:
                    task.cancel()
                
                await asyncio.gather(*pending, return_exceptions=True)
                
                # Close the loop
                loop.run_until_complete(loop.shutdown_asyncgens())
