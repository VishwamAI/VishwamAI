import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union
import json
from pathlib import Path
from .model import VishwamaiModel, VishwamaiConfig

def init_model(
    config: Union[VishwamaiConfig, Dict[str, Any], str, Path],
    pretrained_path: Optional[Union[str, Path]] = None,
    device: str = "cuda",
) -> VishwamaiModel:
    """
    Initialize a Vishwamai model from config and optionally load pretrained weights.
    
    Args:
        config: Model configuration (VishwamaiConfig object, dict, or path to config file)
        pretrained_path: Optional path to pretrained model weights
        device: Device to load the model on
        
    Returns:
        Initialized VishwamaiModel
    """
    # Handle different config input types
    if isinstance(config, (str, Path)):
        with open(config, 'r') as f:
            config_dict = json.load(f)
            config = VishwamaiConfig(**config_dict)
    elif isinstance(config, dict):
        config = VishwamaiConfig(**config)
        
    # Initialize model
    model = VishwamaiModel(config)
    
    # Load pretrained weights if provided
    if pretrained_path is not None:
        pretrained_path = Path(pretrained_path)
        if pretrained_path.is_dir():
            # Load from directory (assumed to be HF format)
            model_path = pretrained_path / "pytorch_model.bin"
            if not model_path.exists():
                model_path = pretrained_path / "model.pt"
        else:
            model_path = pretrained_path
            
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        
    model = model.to(device)
    return model

def create_config_from_template(
    template_name: str,
    output_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> VishwamaiConfig:
    """
    Create a config based on a template with optional modifications.
    
    Args:
        template_name: Name of template config ("2b", "16b", "23b")
        output_path: Optional path to save the config
        **kwargs: Parameters to override in the template
        
    Returns:
        VishwamaiConfig object
    """
    template_configs = {
        "2b": "config_2b.json",
        "16b": "config_16b.json",
        "23b": "config_23.json"
    }
    
    if template_name not in template_configs:
        raise ValueError(f"Unknown template: {template_name}. Available: {list(template_configs.keys())}")
        
    # Load template config
    config_path = Path(__file__).parent / "configs" / template_configs[template_name]
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
        
    # Update with provided parameters
    config_dict.update(kwargs)
    
    # Create config object
    config = VishwamaiConfig(**config_dict)
    
    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
            
    return config

def load_checkpoint(
    path: Union[str, Path],
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Load a complete checkpoint including model, tokenizer and training state.
    
    Args:
        path: Path to checkpoint directory
        device: Device to load the model on
        
    Returns:
        Dictionary containing model, tokenizer, and training state
    """
    path = Path(path)
    
    # Load config and create model
    with open(path / "config.json", 'r') as f:
        config = VishwamaiConfig(**json.load(f))
    model = VishwamaiModel(config)
    
    # Load model weights
    model_path = path / "model.pt"
    if model_path.exists():
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
    
    model = model.to(device)
    
    # Load training state if exists
    training_state = {}
    training_state_path = path / "training_state.pt"
    if training_state_path.exists():
        training_state = torch.load(training_state_path, map_location="cpu")
        
    return {
        "model": model,
        "config": config,
        "training_state": training_state
    }
