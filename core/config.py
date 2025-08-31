import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    """Configuration class for TTS Provider"""
    
    # Server configuration
    HOST = os.environ.get("TTS_HOST", "0.0.0.0")
    PORT = int(os.environ.get("TTS_PORT", 9000))
    
    # Hugging Face configuration
    HF_TOKEN = os.environ.get("HF_TOKEN", None)
    HF_HOME = os.environ.get("HF_HOME", "/app/huggingface_cache")
    
    # GPU configuration
    NVIDIA_VISIBLE_DEVICES = os.environ.get("NVIDIA_VISIBLE_DEVICES", "all")
    NVIDIA_DRIVER_CAPABILITIES = os.environ.get("NVIDIA_DRIVER_CAPABILITIES", "compute,utility")
    
    # Model configuration
    DEFAULT_MODEL = "edge"  # Default model if not specified by client
    
    # Logging configuration
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")