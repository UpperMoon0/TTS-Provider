import os
import logging
from typing import Optional
from huggingface_hub import HfFolder, snapshot_download

class ModelLoader:
    """
    Handles loading of AI models with support for both downloading and reusing existing model files.
    """
    
    def __init__(self, logger=None):
        """Initialize the model loader."""
        self.logger = logger or logging.getLogger("ModelLoader")
    
    def _configure_hf_token(self) -> None:
        """Configure the HuggingFace token from environment variables."""
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            HfFolder.save_token(hf_token)
        else:
            self.logger.warning("HF_TOKEN not set. Some models may not be accessible.")
        return hf_token

    def get_model_path(self, mode: str = "download", model_path: Optional[str] = None) -> Optional[str]:
        """
        Get the path to the model files, either by downloading or using existing files.
        
        Args:
            mode: Either "download" to use HuggingFace cache or "reuse" to use existing files
            model_path: Path to existing model folder when mode is "reuse"
                       Example: "D:/Dev/Models/huggingface/hub/models--sesame--csm-1b"
                       
        Returns:
            Path to the model directory or None if there was an error
        """
        if mode not in ["download", "reuse"]:
            self.logger.error(f"Invalid mode: {mode}. Must be 'download' or 'reuse'")
            return None
            
        if mode == "reuse":
            if not model_path:
                self.logger.error("model_path must be provided when mode is 'reuse'")
                return None
                
            if not os.path.exists(model_path):
                self.logger.error(f"Model path does not exist: {model_path}")
                return None
                
            # Check for required model folders
            required_folders = ["blobs", "refs", "snapshots"]
            missing = [f for f in required_folders if not os.path.exists(os.path.join(model_path, f))]
            if missing:
                self.logger.error(f"Model path is missing required folders: {', '.join(missing)}")
                return None
                
            return model_path
            
        else:  # mode == "download"
            # Configure HF token and download model
            hf_token = self._configure_hf_token()
            if not hf_token:
                self.logger.error("HF_TOKEN not set. Cannot download model.")
                return None
                
            try:
                return snapshot_download("sesame/csm-1b", token=hf_token)
            except Exception as e:
                self.logger.error(f"Error downloading model: {str(e)}")
                return None