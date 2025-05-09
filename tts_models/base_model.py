from abc import ABC, abstractmethod
import asyncio # Add asyncio import
import logging
from typing import Dict # Add Dict for type hinting
import websockets # Required for keep-alive type hints and exceptions

class BaseTTSModel(ABC):
    """Base class for all TTS models"""

    def __init__(self):
        # Initialize logger for the concrete class name
        self.logger = logging.getLogger(self.__class__.__name__)

    async def _run_blocking_task(self, func, *args, **kwargs):
        """Helper to run a synchronous function in a separate thread."""
        return await asyncio.to_thread(func, *args, **kwargs)

    async def run_task_with_keepalive(self, websocket, task_to_run_coro, ping_interval=15):
        """
        Runs a given task coroutine while sending keep-alive pings over the websocket.
        """
        if not websocket:
            # If no websocket is provided, just run the task
            return await task_to_run_coro

        ping_task = None
        # Use a shared event to signal the ping loop to stop
        stop_pinging_event = asyncio.Event()

        async def _ping_loop():
            try:
                while not stop_pinging_event.is_set():
                    try:
                        if websocket.closed:
                            self.logger.info(f"WebSocket connection closed ({websocket.remote_address}), stopping keep-alive pings.")
                            break
                        await websocket.ping()
                        self.logger.debug(f"Sent keep-alive ping to {websocket.remote_address}.")
                    except websockets.exceptions.ConnectionClosed:
                        self.logger.info(f"Connection closed during keep-alive ping to {websocket.remote_address}.")
                        break # Exit loop if connection is closed
                    except Exception as e:
                        self.logger.warning(f"Error sending keep-alive ping to {websocket.remote_address}: {e}")
                        # Depending on the error, you might want to break or continue
                        # For now, we'll let it try again after the interval
                    
                    try:
                        # Wait for the ping interval or until the stop event is set
                        await asyncio.wait_for(asyncio.shield(stop_pinging_event.wait()), timeout=ping_interval)
                    except asyncio.TimeoutError:
                        # Timeout means it's time for the next ping
                        continue
                    except asyncio.CancelledError:
                        break # Loop cancelled
            except asyncio.CancelledError:
                self.logger.debug(f"Keep-alive ping loop for {websocket.remote_address} was cancelled.")
            finally:
                self.logger.debug(f"Keep-alive ping loop for {websocket.remote_address} finished.")
        
        ping_task = asyncio.create_task(_ping_loop())
        
        try:
            # Run the main task
            result = await task_to_run_coro
            return result
        finally:
            # Signal the ping loop to stop and wait for it to finish
            if ping_task:
                stop_pinging_event.set()
                try:
                    # Give it time to exit gracefully, ensure it's not awaited if already done
                    if not ping_task.done():
                        await asyncio.wait_for(ping_task, timeout=ping_interval + 5) 
                except asyncio.TimeoutError:
                    self.logger.warning(f"Keep-alive ping task for {websocket.remote_address} did not finish in time, cancelling.")
                    if not ping_task.done():
                        ping_task.cancel()
                except asyncio.CancelledError:
                     self.logger.debug(f"Keep-alive ping task for {websocket.remote_address} was already cancelled during cleanup.")
                except Exception as e:
                    self.logger.warning(f"Error during ping_task cleanup for {websocket.remote_address}: {e}")
    
    @abstractmethod
    async def generate_speech(self, text: str, speaker: int = 0, lang: str = "en-US", websocket=None, **kwargs) -> bytes:
        """
        Generate speech from text
        
        Args:
            text: Text to convert to speech
            speaker: Speaker ID
            lang: Language code (e.g., "en-US", "ja-JP")
            websocket: Optional websocket connection for keep-alive pings
            # max_audio_length_ms: Removed
            **kwargs: Additional model-specific parameters
            
        Returns:
            Audio bytes in WAV format
        """
        pass
    
    @abstractmethod
    def get_sample_rate(self) -> int:
        """
        Get the sample rate of the generated audio
        
        Returns:
            Sample rate in Hz
        """
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """
        Check if the model is ready to generate speech
        
        Returns:
            True if the model is ready, False otherwise
        """
        pass
    
    @abstractmethod
    async def load(self, websocket=None) -> bool:
        """
        Load the model. Can optionally send keep-alive pings over the websocket.
        
        Returns:
            True if the model was loaded successfully, False otherwise
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Get the name of the model
        
        Returns:
            Model name
        """
        pass
    
    @property
    @abstractmethod
    def supported_speakers(self) -> Dict[int, str]:
        """
        Get the supported speakers, typically for a default or primary language.
        This might be derived from `supported_languages_and_voices` in concrete classes
        or could be deprecated in the future.
        
        Returns:
            Dict mapping speaker IDs (int) to descriptions (str)
        """
        pass

    @property
    @abstractmethod
    def supported_languages_and_voices(self) -> Dict[str, Dict[int, str]]:
        """
        Get all supported languages and the voices/speakers available for each.
        
        Returns:
            A dictionary where:
                - Keys are language codes (e.g., "en-US", "ja-JP").
                - Values are dictionaries mapping speaker IDs (int) to speaker descriptions (str).
                Example: {"en-US": {0: "Male", 1: "Female"}, "ja-JP": {0: "Standard"}}
        """
        pass
