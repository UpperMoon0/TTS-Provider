import os
import logging
from typing import Optional
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
                # Increased timeouts and added retry attempts
                snapshot_download(
                    "sesame/csm-1b",
                    local_dir=model_path,
                    local_dir_use_symlinks=False,
                    token=os.getenv("HUGGINGFACE_TOKEN"),  # Use token if available
                    max_retries=5,
                    force_download=True,
                    resume_download=True,
                    proxies=None,
                    etag_timeout=100,
                    local_files_only=False,
                    token_timeout=100
                )
                self.logger.info(f"Model downloaded successfully to {model_path}")
            except LocalTokenNotFoundError:
                self.logger.warning("No Hugging Face token found. Attempting anonymous download...")
                # Retry without token
                snapshot_download(
                    "sesame/csm-1b",
                    local_dir=model_path,
                    local_dir_use_symlinks=False,
                    max_retries=5,
                    force_download=True,
                    resume_download=True,
                    proxies=None,
                    etag_timeout=100,
                    local_files_only=False
                )
                self.logger.info(f"Model downloaded successfully to {model_path}")
            except Exception as e:
                self.logger.error(f"Error downloading model: {str(e)}")
                raise RuntimeError(f"Failed to download model: {str(e)}")
        
        return model_path