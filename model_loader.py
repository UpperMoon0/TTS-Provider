import os
import logging
import sys
import importlib.util
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
        self.csm_generator = None
    
    def get_model_path(self) -> str:
        """
        Get or create the path to the model files. Downloads if not present.
        
        Returns:
            Path to the model directory
        """
        model_path = os.path.join(self.base_path, "csm-1b")
        
        # If model doesn't exist, download it
        if not os.path.exists(model_path) or not os.listdir(model_path):
            self.logger.info("Model not found locally, downloading...")
            try:
                # Updated parameters to remove deprecated ones
                snapshot_download(
                    repo_id="sesame/csm-1b",
                    local_dir=model_path,
                    token=os.getenv("HUGGINGFACE_TOKEN"),  # Use token if available
                    force_download=True,  # Force download instead of using resume_download
                    proxies=None,
                    etag_timeout=100,
                    local_files_only=False
                )
                self.logger.info(f"Model downloaded successfully to {model_path}")
            except LocalTokenNotFoundError:
                self.logger.warning("No Hugging Face token found. Attempting anonymous download...")
                # Retry without token
                snapshot_download(
                    repo_id="sesame/csm-1b",
                    local_dir=model_path,
                    force_download=True,
                    proxies=None,
                    etag_timeout=100,
                    local_files_only=False
                )
                self.logger.info(f"Model downloaded successfully to {model_path}")
            except Exception as e:
                self.logger.error(f"Error downloading model: {str(e)}")
                raise RuntimeError(f"Failed to download model: {str(e)}")
        
        return model_path
    
    def setup_csm_imports(self):
        """
        Setup the Python path to properly import from Sesame-CSM-1b-Impl
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
        try:
            # Create a simple package structure mapping
            # This makes Python think 'csm' is a real module that points to our implementation
            csm_impl_path = Path(__file__).parent / "Sesame-CSM-1b-Impl"
            if not csm_impl_path.exists():
                self.logger.error(f"CSM implementation not found at {csm_impl_path}")
                return False
            
            self.logger.info(f"Setting up imports for CSM implementation at {csm_impl_path}")
            
            # Add the implementation directory to Python path
            sys.path.insert(0, str(csm_impl_path))
            
            # Create csm namespace package if it doesn't exist
            if 'csm' not in sys.modules:
                # Create an empty module object for csm
                csm_module = type('module', (), {})
                csm_module.__path__ = []
                sys.modules['csm'] = csm_module
            
            # Create csm.models that points to our models.py
            models_module = importlib.import_module('models')
            sys.modules['csm.models'] = models_module
            
            # Create csm.watermarking that points to our watermarking.py
            watermarking_module = importlib.import_module('watermarking')
            sys.modules['csm.watermarking'] = watermarking_module
            
            # Add modules to the csm module
            sys.modules['csm'].models = models_module
            sys.modules['csm'].watermarking = watermarking_module
            
            self.logger.info("CSM import setup successful")
            return True
        except Exception as e:
            self.logger.error(f"Failed to setup CSM imports: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def load_csm_model(self):
        """
        Load the CSM-1B model.
        
        Returns:
            Loaded CSM generator model instance or None if loading failed
        """
        try:
            # First, set up the import system for CSM modules
            if not self.setup_csm_imports():
                return None
            
            # Now we can properly import from the generator module
            try:
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
                self.logger.error(f"Failed to import from Sesame-CSM-1b-Impl: {str(e)}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to load CSM-1B model: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None