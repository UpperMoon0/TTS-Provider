import torch
import logging

"""
This module patches silentcipher to use torch.load with weights_only=True
to avoid FutureWarning messages in newer versions of PyTorch.
"""

# Store original torch.load function
original_torch_load = torch.load
# Track if patch is already applied
is_patch_applied = False
logger = logging.getLogger("silentcipher_patch")

def patched_torch_load(*args, **kwargs):
    """
    A patched version of torch.load that sets weights_only=True by default
    to prevent FutureWarning messages.
    """
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = True
    return original_torch_load(*args, **kwargs)

def apply_patch():
    """
    Apply the patch to torch.load in silentcipher module.
    Must be called before any silentcipher functions are used.
    Only applies the patch if it hasn't been applied already.
    """
    global is_patch_applied
    
    # Only apply patch if not already applied
    if not is_patch_applied:
        # Set torch.load to our patched version
        torch.load = patched_torch_load
        is_patch_applied = True
        
        # Log successful patching with logger
        logger.info("Patched silentcipher to use torch.load with weights_only=True")
        return True
    else:
        logger.debug("Patch already applied, skipping")
        return False

def remove_patch():
    """
    Remove the patch and restore the original torch.load function.
    Only removes the patch if it was previously applied.
    """
    global is_patch_applied
    
    if is_patch_applied:
        torch.load = original_torch_load
        is_patch_applied = False
        logger.debug("Removed torch.load patch")
        return True
    else:
        logger.debug("Patch not applied, nothing to remove")
        return False
