import logging
from typing import Dict, List, Optional, Type
from .base_model import BaseTTSModel
from .edge_tts import EdgeTTSModel

# Dictionary to store already-initialized model instances
_model_instances = {}

class TTSModelFactory:
    """Factory class to create TTS models"""
    
    # Available model identifiers and their implementation classes
    # Note: We only define the class references here, instances are created on demand
    AVAILABLE_MODELS = {
        "sesame": "SesameCSMModel",  # Store as string to avoid immediate import
        "csm": "SesameCSMModel",      # Alias
        "edge": EdgeTTSModel,         # Edge TTS is lightweight, so we can import directly
        "edge-tts": EdgeTTSModel,     # Alias
        "zonos": "ZonosTTSModel",     # Add Zonos, store as string for on-demand import
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
        model_class = cls.AVAILABLE_MODELS.get(model_name.lower())
        
        # If the model class is stored as a string, it needs to be imported
        if isinstance(model_class, str):
            logger = logging.getLogger("TTSModelFactory")
            if model_class == "SesameCSMModel":
                try:
                    logger.info("Importing SesameCSMModel on demand")
                    from .sesame_csm import SesameCSMModel
                    # Update the dictionary with the actual class for future lookups
                    cls.AVAILABLE_MODELS["sesame"] = SesameCSMModel # type: ignore
                    cls.AVAILABLE_MODELS["csm"] = SesameCSMModel # type: ignore
                    return SesameCSMModel
                except ImportError as e:
                    logger.error(f"Failed to import SesameCSMModel: {e}")
                    return None
            elif model_class == "ZonosTTSModel":
                try:
                    logger.info("Importing ZonosTTSModel on demand")
                    from .zonos_tts import ZonosTTSModel
                    # Update the dictionary with the actual class for future lookups
                    cls.AVAILABLE_MODELS["zonos"] = ZonosTTSModel # type: ignore
                    return ZonosTTSModel
                except ImportError as e:
                    logger.error(f"Failed to import ZonosTTSModel: {e}")
                    return None
        
        return model_class
    
    @classmethod
    def create_model(cls, model_name: str) -> Optional[BaseTTSModel]:
        """
        Create a model instance for the given model name
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model instance or None if not found
        """
        global _model_instances
        model_key = model_name.lower()
        
        # Check if we already have an instance of this model
        if model_key in _model_instances:
            return _model_instances[model_key]
        
        # Get the model class
        model_class = cls.get_model_class(model_key)
        if model_class:
            logger = logging.getLogger("TTSModelFactory")
            logger.info(f"Creating model instance for {model_name}")
            
            try:
                # Create the model instance
                model_instance = model_class()
                # Store the instance for reuse
                _model_instances[model_key] = model_instance
                return model_instance
            except Exception as e:
                logger.error(f"Error creating model instance for {model_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
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
