# filepath: d:\Dev\Workspace\Python\AI-Content-Generation\TTS-Provider\tests\logging_utils.py
"""
Utilities for handling logging in tests to prevent 'I/O operation on closed file' errors.
"""

import logging
import sys
from contextlib import contextmanager

# Create a NullHandler that doesn't actually do any I/O
class SafeHandler(logging.Handler):
    """A handler that doesn't raise exceptions during interpreter shutdown."""
    
    def emit(self, record):
        try:
            msg = self.format(record)
            # Only print critical errors to stderr
            if record.levelno >= logging.CRITICAL:
                print(msg, file=sys.stderr)
        except (ValueError, IOError):
            # Ignore I/O errors during shutdown
            pass
        except Exception:
            # Ignore any other errors during shutdown
            pass


def configure_safe_logging():
    """Configure logging to be safe during interpreter shutdown."""
    # Remove any existing handlers
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    
    # Add our safe handler
    safe_handler = SafeHandler()
    safe_handler.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    safe_handler.setFormatter(formatter)
    root_logger.addHandler(safe_handler)
    
    # Also configure the asyncio logger specifically
    asyncio_logger = logging.getLogger('asyncio')
    asyncio_logger.setLevel(logging.ERROR)
    
    # And the websockets logger
    websockets_logger = logging.getLogger('websockets')
    websockets_logger.setLevel(logging.ERROR)
    
    return safe_handler


@contextmanager
def suppress_logging_errors():
    """Context manager to suppress logging errors during test teardown."""
    # Save original excepthook
    original_excepthook = sys.excepthook
    
    # Define a custom excepthook that ignores ValueError from logging
    def custom_excepthook(exc_type, exc_value, exc_traceback):
        if exc_type is ValueError and "I/O operation on closed file" in str(exc_value):
            # Silently ignore this specific error
            return
        # For other exceptions, use the original excepthook
        original_excepthook(exc_type, exc_value, exc_traceback)
    
    # Set our custom excepthook
    sys.excepthook = custom_excepthook
    
    try:
        yield
    finally:
        # Restore original excepthook
        sys.excepthook = original_excepthook