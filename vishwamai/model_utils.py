import os
import json
from typing import Optional

import torch
import torch.distributed as dist

from .model import Transformer, ModelArgs

def setup_distributed():
    """Setup distributed training if available."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        return True
    return False

def get_gpu_memory():
    """Get available GPU memory in GB."""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        return total_memory / (1024**3)  # Convert to GB
    return 0

def optimize_config_for_gpu(config_path: str, gpu_memory: float):
    """Optimize model configuration based on available GPU memory."""
    with open(config_path) as f:
        config = json.load(f)
    
    # Adjust model size based on GPU memory
    if gpu_memory < 8:  # Less than 8GB (T4)
        config.update({
            'dim': 1024,
            'inter_dim': 4096,
            'n_heads': 8,
            'batch_size': 1,
            'gradient_accumulation_steps': 16,
            'max_seq_len': 1024
        })
    elif gpu_memory < 16:  # Less than 16GB (P100, older GPUs)
        config.update({
            'dim': 2048,
            'inter_dim': 8192,
            'n_heads': 16,
            'batch_size': 2,
            'gradient_accumulation_steps': 8,
            'max_seq_len': 2048
        })
    else:  # 16GB or more (V100, A100)
        config.update({
            'dim': 2048,
            'inter_dim': 10944,
            'n_heads': 16,
            'batch_size': 4,
            'gradient_accumulation_steps': 4,
            'max_seq_len': 4096
        })
    
    # Enable performance optimizations
    config.update({
        'use_flash_attention': True,
        'gradient_checkpointing': True,
        'fp16': True if gpu_memory < 32 else False,  # Use FP16 for smaller GPUs
        'bf16': True if gpu_memory >= 32 else False  # Use BF16 for A100
    })
    
    return config

def load_model(
    config_path: str,
    device: str = "cuda",
    pretrained_path: Optional[str] = None,
    use_cache: bool = True
) -> Transformer:
    """Load VishwamAI model with optimized settings."""
    
    # Get GPU memory and optimize config
    gpu_memory = get_gpu_memory()
    config = optimize_config_for_gpu(config_path, gpu_memory)
    
    # Initialize model args
    model_args = ModelArgs(
        max_batch_size=config['batch_size'],
        max_seq_len=config['max_seq_len'],
        dim=config['dim'],
        inter_dim=config['inter_dim'],
        n_heads=config['n_heads'],
        dtype="fp8" if gpu_memory >= 32 else "bf16"  # Use FP8 for larger GPUs
    )
    
    # Create model
    model = Transformer(model_args)
    
    if pretrained_path and os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
    # Setup distributed training
    if setup_distributed():
        model = torch.nn.parallel.DistributedDataParallel(model)
    
    # Move to device and set training mode
    model = model.to(device)
    model.train()
    
    if not use_cache:
        model.config.use_cache = False
    
    # Enable memory optimizations
    if config.get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
    
    return model

def get_training_config(model_args: ModelArgs, gpu_memory: float):
    """Get optimized training configuration."""
    return {
        'learning_rate': 2e-5 if gpu_memory < 16 else 3e-5,
        'warmup_steps': 100,
        'max_steps': 1000,
        'eval_steps': 100,
        'save_steps': 200,
        'weight_decay': 0.01,
        'logging_steps': 10,
        'fp16': gpu_memory < 32,
        'bf16': gpu_memory >= 32,
        'gradient_checkpointing': True,
        'evaluation_strategy': 'steps',
        'save_strategy': 'steps'
    }
