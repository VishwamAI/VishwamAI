import os
import json
import torch
from typing import Optional
from .model import Transformer, ModelArgs

def load_model(
    config_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    pretrained_path: Optional[str] = None,
    use_cache: bool = True
) -> Transformer:
    """Load VishwamAI model with configuration."""
    
    # Load configuration
    with open(config_path) as f:
        config = json.load(f)
    
    # Create model arguments
    model_args = ModelArgs(
        max_batch_size=config.get("max_batch_size", 8),
        max_seq_len=config.get("max_seq_len", 4096 * 4),
        dtype=config.get("dtype", "bf16"),
        vocab_size=config.get("vocab_size", 102400),
        dim=config.get("dim", 2048),
        inter_dim=config.get("inter_dim", 10944),
        moe_inter_dim=config.get("moe_inter_dim", 1408),
        n_layers=config.get("n_layers", 27),
        n_dense_layers=config.get("n_dense_layers", 1),
        n_heads=config.get("n_heads", 16),
        n_routed_experts=config.get("n_routed_experts", 64),
        n_shared_experts=config.get("n_shared_experts", 2),
        n_activated_experts=config.get("n_activated_experts", 6),
        n_expert_groups=config.get("n_expert_groups", 1),
        n_limited_groups=config.get("n_limited_groups", 1),
        score_func=config.get("score_func", "softmax"),
        route_scale=config.get("route_scale", 1.0),
        q_lora_rank=config.get("q_lora_rank", 0),
        kv_lora_rank=config.get("kv_lora_rank", 512),
        qk_nope_head_dim=config.get("qk_nope_head_dim", 128),
        qk_rope_head_dim=config.get("qk_rope_head_dim", 64),
        v_head_dim=config.get("v_head_dim", 128),
        original_seq_len=config.get("original_seq_len", 4096),
        rope_theta=config.get("rope_theta", 10000.0),
        rope_factor=config.get("rope_factor", 40),
        beta_fast=config.get("beta_fast", 32),
        beta_slow=config.get("beta_slow", 1),
        mscale=config.get("mscale", 1.0)
    )
    
    # Initialize model
    model = Transformer(model_args)
    
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

def find_optimal_batch_size(
    model: Transformer,
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
                0, model.embed.vocab_size,
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
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, save_path)

def load_checkpoint(
    checkpoint_path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> tuple:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, checkpoint['epoch'], checkpoint['loss']
