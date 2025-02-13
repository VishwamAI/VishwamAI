import json
from pathlib import Path
from typing import Optional, Union
import torch
from .model import Transformer, ModelArgs

def load_config(config_path: Union[str, Path]) -> dict:
    """Load a model configuration file"""
    with open(config_path) as f:
        config = json.load(f)
    return config

def config_to_model_args(config: dict) -> ModelArgs:
    """Convert a configuration dictionary to ModelArgs"""
    args = ModelArgs()
    
    # Basic model configuration
    args.vocab_size = config.get('vocab_size', args.vocab_size)
    args.dim = config.get('hidden_size', args.dim)
    args.inter_dim = config.get('intermediate_size', args.inter_dim)
    args.n_heads = config.get('num_attention_heads', args.n_heads)
    args.n_layers = config.get('num_hidden_layers', args.n_layers)
    args.max_seq_len = config.get('max_position_embeddings', args.max_seq_len)
    
    # MoE configuration
    if 'moe_config' in config:
        moe_config = config['moe_config']
        args.n_routed_experts = moe_config.get('num_experts', args.n_routed_experts)
        args.n_activated_experts = moe_config.get('num_experts_per_tok', args.n_activated_experts)
    
    # Cache augmentation configuration
    if 'cache_augmentation' in config:
        cache_config = config['cache_augmentation']
        args.use_cache_augmentation = cache_config.get('enabled', True)
        args.cache_hidden_size = cache_config.get('hidden_size', 256)
        args.cache_num_heads = cache_config.get('num_heads', 4)
        args.cache_dropout = cache_config.get('dropout', 0.1)
        args.cache_max_length = cache_config.get('max_length', 1024)
    
    # Neural memory configuration
    if 'neural_memory' in config:
        memory_config = config['neural_memory']
        args.use_neural_memory = memory_config.get('enabled', True)
        args.memory_size = memory_config.get('memory_size', 512)
        args.num_memory_layers = memory_config.get('num_memory_layers', 3)
        args.memory_dropout = memory_config.get('dropout', 0.1)
    
    # Tree of thoughts configuration
    if 'tree_of_thoughts' in config:
        tot_config = config['tree_of_thoughts']
        args.use_tree_of_thoughts = tot_config.get('enabled', True)
        args.tot_max_branches = tot_config.get('max_branches', 4)
        args.tot_max_depth = tot_config.get('max_depth', 3)
        args.tot_beam_width = tot_config.get('beam_width', 2)
        args.tot_temperature = tot_config.get('temperature', 0.8)
        args.tot_min_score_diff = tot_config.get('min_score_diff', 0.1)
    
    # RoPE configuration
    if 'rope_scaling' in config:
        rope_config = config['rope_scaling']
        args.rope_factor = rope_config.get('factor', args.rope_factor)
    
    # Training precision
    args.dtype = "fp8" if config.get('use_fp8', False) else "bf16"
    
    return args

def load_model(
    config_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    **kwargs
) -> Transformer:
    """
    Load a Transformer model from a configuration file.
    
    Args:
        config_path: Path to the configuration file
        device: Device to load the model on
        **kwargs: Additional arguments to override configuration values
    
    Returns:
        Initialized Transformer model
    """
    # Load configuration
    config = load_config(config_path)
    
    # Override config with kwargs
    for k, v in kwargs.items():
        if '.' in k:
            # Handle nested config values
            parts = k.split('.')
            curr = config
            for part in parts[:-1]:
                if part not in curr:
                    curr[part] = {}
                curr = curr[part]
            curr[parts[-1]] = v
        else:
            config[k] = v
    
    # Convert to model args
    args = config_to_model_args(config)
    
    # Initialize model
    model = Transformer(args)
    
    # Move to device if specified
    if device is not None:
        model = model.to(device)
    
    return model

def get_model_size(model: Transformer) -> int:
    """Calculate the total number of parameters in the model"""
    return sum(p.numel() for p in model.parameters())

def print_model_size(model: Transformer) -> None:
    """Print the model size in a human readable format"""
    num_params = get_model_size(model)
    
    if num_params < 1e6:
        print(f"Model size: {num_params:,} parameters")
    elif num_params < 1e9:
        print(f"Model size: {num_params/1e6:.1f}M parameters")
    else:
        print(f"Model size: {num_params/1e9:.1f}B parameters")

def save_pretrained(
    model: Transformer,
    save_directory: Union[str, Path],
    save_config: bool = True
) -> None:
    """
    Save a model to a directory, optionally saving its configuration
    
    Args:
        model: The model to save
        save_directory: Directory to save the model to
        save_config: Whether to save the model's configuration
    """
    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)
    
    # Save the model weights
    model_path = save_directory / "pytorch_model.bin"
    torch.save(model.state_dict(), model_path)
    
    # Save the configuration if requested
    if save_config:
        config = {
            "model_type": "vishwamai",
            "architectures": ["VishwamaiModel"],
            "vocab_size": model.args.vocab_size,
            "hidden_size": model.args.dim,
            "intermediate_size": model.args.inter_dim,
            "num_attention_heads": model.args.n_heads,
            "num_hidden_layers": model.args.n_layers,
            "max_position_embeddings": model.args.max_seq_len,
            "use_cache": True,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            
            # Advanced features
            "cache_augmentation": {
                "enabled": model.args.use_cache_augmentation,
                "hidden_size": model.args.cache_hidden_size,
                "num_heads": model.args.cache_num_heads,
                "dropout": model.args.cache_dropout,
                "max_length": model.args.cache_max_length
            },
            "neural_memory": {
                "enabled": model.args.use_neural_memory,
                "memory_size": model.args.memory_size,
                "num_memory_layers": model.args.num_memory_layers,
                "dropout": model.args.memory_dropout
            },
            "tree_of_thoughts": {
                "enabled": model.args.use_tree_of_thoughts,
                "max_branches": model.args.tot_max_branches,
                "max_depth": model.args.tot_max_depth,
                "beam_width": model.args.tot_beam_width,
                "temperature": model.args.tot_temperature,
                "min_score_diff": model.args.tot_min_score_diff
            }
        }
        
        config_path = save_directory / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

def load_pretrained(
    model_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    **kwargs
) -> Transformer:
    """
    Load a pretrained model from a directory
    
    Args:
        model_path: Path to the model directory
        device: Device to load the model on
        **kwargs: Additional arguments to override configuration values
    
    Returns:
        Loaded Transformer model
    """
    model_path = Path(model_path)
    
    # Load configuration
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise ValueError(f"No configuration file found at {config_path}")
    
    # Initialize model
    model = load_model(config_path, device, **kwargs)
    
    # Load weights
    weights_path = model_path / "pytorch_model.bin"
    if not weights_path.exists():
        raise ValueError(f"No model weights found at {weights_path}")
    
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    
    if device is not None:
        model = model.to(device)
    
    return model
