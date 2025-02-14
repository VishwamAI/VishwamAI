import os
import json
import torch
from typing import Optional
from .model import VishwamAIModel, ModelArgs

def load_model(
    config_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    pretrained_path: Optional[str] = None,
    use_cache: bool = True
) -> VishwamAIModel:
    """Load VishwamAI model with configuration."""
    
    # Load configuration
    with open(config_path) as f:
        config = json.load(f)
    
    # Create model arguments
    model_args = ModelArgs(
        dim=config.get("dim", 8192),
        n_layers=config.get("n_layers", 120),
        vocab_size=config.get("vocab_size", 64000),
        max_seq_len=config.get("max_seq_len", 32768),
        num_attention_heads=config.get("num_attention_heads", 64),
        use_neural_memory=config.get("use_neural_memory", True),
        use_tree_of_thoughts=config.get("use_tree_of_thoughts", True),
        use_cache_augmentation=config.get("use_cache_augmentation", True),
        memory_size=config.get("memory_size", 2048),
        tree_beam_width=config.get("tree_beam_width", 4),
        cache_size=config.get("cache_size", 65536)
    )
    
    # Initialize model
    model = VishwamAIModel(model_args)
    
    # Load pretrained weights if available
    if pretrained_path and os.path.exists(pretrained_path):
        state_dict = torch.load(
            pretrained_path,
            map_location=device
        )
        model.load_state_dict(state_dict)
    
    model.to(device)
    return model

def get_gpu_memory() -> float:
    """Get available GPU memory in GB."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    return 0.0

def enable_memory_efficient_attention(model: VishwamAIModel):
    """Enable memory efficient attention for large models."""
    for block in model.blocks:
        if hasattr(block.attention, "enable_memory_efficient_attention"):
            block.attention.enable_memory_efficient_attention()

def find_optimal_batch_size(
    model: VishwamAIModel,
    starting_batch_size: int = 8,
    gpu_memory_threshold: float = 0.9,
    sequence_length: int = 2048
) -> int:
    """Find optimal batch size for given model and GPU memory."""
    if not torch.cuda.is_available():
        return 1
        
    total_memory = get_gpu_memory()
    batch_size = starting_batch_size
    
    while True:
        try:
            # Test batch with random inputs
            inputs = torch.randint(
                0, model.args.vocab_size,
                (batch_size, sequence_length),
                device="cuda"
            )
            
            # Clear cache
            torch.cuda.empty_cache()
            
            # Test forward pass
            with torch.no_grad():
                model(inputs)
            
            # Check memory usage
            memory_used = torch.cuda.memory_allocated() / 1e9
            if memory_used / total_memory > gpu_memory_threshold:
                return batch_size // 2
            
            batch_size *= 2
            
        except RuntimeError:  # Out of memory
            return batch_size // 2

def save_checkpoint(
    model: VishwamAIModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str
):
    """Save model checkpoint with all components."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_args': model.args
    }
    
    # Save components separately
    if hasattr(model, 'memory'):
        checkpoint['memory_state'] = model.memory.state_dict()
    
    if hasattr(model, 'tree'):
        checkpoint['tree_state'] = model.tree.state_dict()
        
    if hasattr(model, 'cache'):
        checkpoint['cache_state'] = model.cache.state_dict()
    
    torch.save(checkpoint, save_path)

def load_checkpoint(
    checkpoint_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> tuple:
    """Load model checkpoint with all components."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = VishwamAIModel(checkpoint['model_args'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load components
    if 'memory_state' in checkpoint and hasattr(model, 'memory'):
        model.memory.load_state_dict(checkpoint['memory_state'])
        
    if 'tree_state' in checkpoint and hasattr(model, 'tree'):
        model.tree.load_state_dict(checkpoint['tree_state'])
        
    if 'cache_state' in checkpoint and hasattr(model, 'cache'):
        model.cache.load_state_dict(checkpoint['cache_state'])
    
    model.to(device)
    
    return model, checkpoint['epoch'], checkpoint['loss']
