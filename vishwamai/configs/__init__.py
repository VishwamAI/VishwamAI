"""Configuration management for VishwamAI."""

import os
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import yaml
from dataclasses import dataclass

CONFIG_TYPES = [
    "model",
    "moe",
    "mla",
    "training",
    "data",
    "tpu"
]

@dataclass
class ConfigManager:
    """Manages loading and validation of configurations."""
    
    config_dir: str
    model_config: Optional[Dict[str, Any]] = None
    moe_config: Optional[Dict[str, Any]] = None
    mla_config: Optional[Dict[str, Any]] = None
    training_config: Optional[Dict[str, Any]] = None
    data_config: Optional[Dict[str, Any]] = None
    tpu_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default config paths."""
        self.config_paths = {
            "model": os.path.join(self.config_dir, "model_config.yaml"),
            "moe": os.path.join(self.config_dir, "moe_config.yaml"),
            "mla": os.path.join(self.config_dir, "mla_config.yaml"),
            "training": os.path.join(self.config_dir, "training_config.yaml"),
            "data": os.path.join(self.config_dir, "data_config.yaml"),
            "tpu": os.path.join(self.config_dir, "tpu_config.yaml")
        }
        
    def load_config(self, config_type: str) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Args:
            config_type: Type of configuration to load
            
        Returns:
            Configuration dictionary
            
        Raises:
            ValueError: If config_type is invalid
        """
        if config_type not in CONFIG_TYPES:
            raise ValueError(f"Invalid config type. Must be one of {CONFIG_TYPES}")
            
        config_path = self.config_paths[config_type]
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        setattr(self, f"{config_type}_config", config)
        return config
        
    def load_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load all configuration files.
        
        Returns:
            Dictionary of all configurations
        """
        configs = {}
        for config_type in CONFIG_TYPES:
            configs[config_type] = self.load_config(config_type)
        return configs
        
    def update_config(self,
                     config_type: str,
                     updates: Dict[str, Any],
                     save: bool = True) -> Dict[str, Any]:
        """Update configuration with new values.
        
        Args:
            config_type: Type of configuration to update
            updates: Dictionary of updates to apply
            save: Whether to save updated config to file
            
        Returns:
            Updated configuration
        """
        config = getattr(self, f"{config_type}_config", None)
        if config is None:
            config = self.load_config(config_type)
            
        # Recursively update nested dictionaries
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
            
        config = update_dict(config, updates)
        setattr(self, f"{config_type}_config", config)
        
        if save:
            self.save_config(config_type)
            
        return config
        
    def save_config(self, config_type: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            config_type: Type of configuration to save
        """
        config = getattr(self, f"{config_type}_config")
        if config is None:
            raise ValueError(f"No {config_type} configuration loaded")
            
        config_path = self.config_paths[config_type]
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
    def merge_configs(self, config_types: List[str]) -> Dict[str, Any]:
        """Merge multiple configurations.
        
        Args:
            config_types: List of configuration types to merge
            
        Returns:
            Merged configuration dictionary
        """
        merged = {}
        for config_type in config_types:
            config = getattr(self, f"{config_type}_config", None)
            if config is None:
                config = self.load_config(config_type)
            merged.update(config)
        return merged
        
    def validate_configs(self) -> List[str]:
        """Validate loaded configurations.
        
        Returns:
            List of validation errors, if any
        """
        errors = []
        
        # Load all configs if not already loaded
        for config_type in CONFIG_TYPES:
            if getattr(self, f"{config_type}_config") is None:
                try:
                    self.load_config(config_type)
                except Exception as e:
                    errors.append(f"Failed to load {config_type} config: {str(e)}")
                    
        # Add config-specific validation here
        # Model config validation
        if self.model_config:
            if self.model_config.get('hidden_size', 0) <= 0:
                errors.append("Model hidden_size must be positive")
                
        # MoE config validation
        if self.moe_config:
            if self.moe_config.get('num_experts', 0) <= 0:
                errors.append("MoE num_experts must be positive")
                
        # MLA config validation
        if self.mla_config:
            if self.mla_config.get('num_prev_layers', 0) <= 0:
                errors.append("MLA num_prev_layers must be positive")
                
        return errors
        
    @property
    def is_valid(self) -> bool:
        """Check if all configurations are valid.
        
        Returns:
            True if all configurations are valid
        """
        return len(self.validate_configs()) == 0

def load_config(config_type: str,
               config_dir: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to load a single configuration.
    
    Args:
        config_type: Type of configuration to load
        config_dir: Optional directory containing config files
        
    Returns:
        Configuration dictionary
    """
    if config_dir is None:
        config_dir = os.path.dirname(os.path.abspath(__file__))
        
    manager = ConfigManager(config_dir)
    return manager.load_config(config_type)

# Global configuration manager instance
CONFIG_MANAGER = ConfigManager(os.path.dirname(os.path.abspath(__file__)))

__all__ = [
    'ConfigManager',
    'CONFIG_MANAGER',
    'load_config',
    'CONFIG_TYPES'
]
