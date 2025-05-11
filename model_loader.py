import importlib
import os
import logging
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalTokenNotFoundError

class ModelLoader:
    """
    Handles loading of AI models with local model storage within the repository.
    """
    
    def __init__(self, logger=None):
        """Initialize the model loader."""
        self.logger = logger or logging.getLogger("ModelLoader")
        self.base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(self.base_path, exist_ok=True)
        # self.csm_generator = None # This attribute doesn't seem to be used.
    
    def get_model_path(self) -> str:
        """
        Get the path to the model files from Hugging Face Hub.
        Downloads to the HF cache if not present.
        
        Returns:
            Path to the model directory in the HF cache.
        """
        self.logger.info("Attempting to load 'sesame/csm-1b' model from Hugging Face cache or download if not present...")
        try:
            # snapshot_download will use the HF cache by default if local_dir is not specified.
            # It returns the path to the downloaded/cached snapshot.
            cached_model_path = snapshot_download(
                repo_id="sesame/csm-1b",
                token=os.getenv("HF_TOKEN"),
                force_download=False,  # Allow caching, do not force re-download
                local_files_only=False, # Allow download if not in cache
                proxies=None,
                etag_timeout=100
            )
            self.logger.info(f"Model 'sesame/csm-1b' successfully located/downloaded to Hugging Face cache: {cached_model_path}")
            return cached_model_path
        except LocalTokenNotFoundError:
            self.logger.warning("No Hugging Face token found for 'sesame/csm-1b'. Attempting anonymous download to cache...")
            # Retry without token
            cached_model_path = snapshot_download(
                repo_id="sesame/csm-1b",
                force_download=False, # Allow caching
                local_files_only=False, # Allow download
                proxies=None,
                etag_timeout=100
            )
            self.logger.info(f"Model 'sesame/csm-1b' successfully located/downloaded to Hugging Face cache (anonymous): {cached_model_path}")
            return cached_model_path
        except Exception as e:
            self.logger.error(f"Error resolving/downloading model 'sesame/csm-1b' from Hugging Face Hub: {str(e)}")
            raise RuntimeError(f"Failed to resolve/download model 'sesame/csm-1b': {str(e)}")
    
    # Removed setup_csm_imports as nstut-csm-fork is now a pip library
    
    def load_csm_model(self):
        """
        Load the CSM-1B model.
        
        Returns:
            Loaded CSM generator model instance or None if loading failed
        """
        try:
            # nstut-csm-fork is now a pip library, so direct import should work.
            # The package structure might be `nstut_csm_fork.generator` or similar.
            # Assuming the import `from generator import load_csm_1b` works if the package is installed correctly.
            try:
                # Attempt to import directly, assuming nstut-csm-fork is in PYTHONPATH via pip install
                from generator import load_csm_1b 
                
                # Get model path
                model_path = self.get_model_path()
                self.logger.info(f"Loading CSM-1B model from {model_path}")
                
                # Import torch here to ensure it's available
                import torch
                
                # Determine device - use CUDA if available
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.logger.info(f"Using device: {device}")
                
                # Load the CSM-1B model
                generator = load_csm_1b(model_path=model_path, device=device)
                self.logger.info("CSM-1B model loaded successfully")
                return generator
                
            except ImportError as e:
                self.logger.error(f"Failed to import 'load_csm_1b' from 'generator' (nstut-csm-fork library): {str(e)}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to load CSM-1B model (nstut-csm-fork library): {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
