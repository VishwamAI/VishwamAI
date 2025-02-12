
from dataclasses import dataclass
from typing import Dict, Any
import json

@dataclass 
class ConfigValidator:
    """Validate and normalize model configurations"""
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate config parameters and set defaults"""
        required_fields = [
            "vocab_size",
            "hidden_size", 
            "num_hidden_layers"
        ]
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")
                
        # Add reasonable defaults
        defaults = {
            "max_position_embeddings": 2048,
            "layer_norm_eps": 1e-5,
            "dropout": 0.1
        }
        
        for k, v in defaults.items():
            config.setdefault(k, v)
            
        return config

    @staticmethod
    def save_validated_config(config: Dict[str, Any], path: str):
        """Save validated config to file"""
        validated = ConfigValidator.validate_config(config)
        with open(path, 'w') as f:
            json.dump(validated, f, indent=2)
