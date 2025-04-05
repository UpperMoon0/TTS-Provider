# filepath: d:\Dev\Workspace\Python\AI-Content-Generation\TTS-Provider\tests\asyncio_helper.py
"""
Helper functions for asyncio testing to avoid 'Task was destroyed but it is pending' warnings.
"""

import asyncio
import logging
import sys

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AsyncHelper")

async def cleanup_pending_tasks():
    """
    Clean up all pending tasks in the current event loop.
    This helps avoid "Task was destroyed but it is pending" warnings.
    """
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    
    if not tasks:
        return
    
    logger.info(f"Cleaning up {len(tasks)} pending tasks")
    
    # Cancel all tasks and suppress exceptions
    for task in tasks:
        if not task.done():
            task.cancel()
    
    # Wait for all tasks to be cancelled
    await asyncio.gather(*tasks, return_exceptions=True)
    
    # Give the event loop a chance to clean up
    await asyncio.sleep(0.1)


def is_windows():
    """Check if the current platform is Windows."""
    return sys.platform.startswith('win')