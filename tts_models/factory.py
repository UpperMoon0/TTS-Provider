import logging
from typing import Dict, List, Optional, Type
from .base_model import BaseTTSModel
from .sesame_csm import SesameCSMModel
from .edge_tts import EdgeTTSModel

class TTSModelFactory:
    """Factory class to create TTS models"""
    
    # Available models with their string identifiers
    AVAILABLE_MODELS = {
        "sesame": SesameCSMModel,
        "csm": SesameCSMModel,  # Alias
        "edge": EdgeTTSModel,
        "edge-tts": EdgeTTSModel,  # Alias
    }
    
    @classmethod
    def get_model_class(cls, model_name: str) -> Optional[Type[BaseTTSModel]]:
        """
        Get the model class for the given model name
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model class or None if not found
        """
        return cls.AVAILABLE_MODELS.get(model_name.lower())
    
    @classmethod
    def create_model(cls, model_name: str) -> Optional[BaseTTSModel]:
        """
        Create a model instance for the given model name
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model instance or None if not found
        """
        model_class = cls.get_model_class(model_name)
        if model_class:
            logger = logging.getLogger("TTSModelFactory")
            logger.info(f"Creating model instance for {model_name}")
            return model_class()
        return None
    
    @classmethod
    def list_available_models(cls) -> List[str]:
        """
        List all available models
        
        Returns:
            List of model names
        """
        # Return unique model names (without aliases)
        return list(set(cls.AVAILABLE_MODELS.values()))
    
    @classmethod
    def get_model_info(cls) -> Dict[str, Dict]:
        """
        Get information about all available models
        
        Returns:
            Dict mapping model names to info dicts
        """
        info = {}
        for model_id, model_class in cls.AVAILABLE_MODELS.items():
            # Create a temporary instance to get information
            model = model_class()
            model_name = model.model_name
            
            if model_name not in info:
                info[model_name] = {
                    "identifiers": [],
                    "speakers": model.supported_speakers,
                    "sample_rate": model.get_sample_rate(),
                }
            
            # Add this identifier to the list
            if model_id not in info[model_name]["identifiers"]:
                info[model_name]["identifiers"].append(model_id)
                
        return info 