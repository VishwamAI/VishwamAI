"""
Training utility functions
"""
import os
import json
import logging
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

def setup_training(
    local_rank: int,
    world_size: int,
    seed: int = 42
) -> None:
    """
    Setup distributed training environment
    
    Args:
        local_rank: Local GPU rank
        world_size: Total number of GPUs
        seed: Random seed
    """
    # Set random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Initialize distributed backend
    if world_size > 1:
        if not dist.is_initialized():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend="nccl",
                world_size=world_size,
                rank=local_rank
            )
            
def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

def log_metrics(
    metrics: Dict[str, float],
    step: int,
    logger: logging.Logger
) -> None:
    """
    Log training metrics
    
    Args:
        metrics: Dictionary of metric names and values
        step: Current training step
        logger: Logger instance
    """
    log_str = f"Step {step} |"
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            log_str += f" {name}: {value:.4f} |"
    logger.info(log_str)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    config: Dict[str, Any],
    metrics: Dict[str, float],
    step: int,
    save_dir: str,
    filename: str = "checkpoint.pt"
) -> None:
    """
    Save training checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Optional scheduler state
        scaler: Optional gradient scaler state
        config: Training configuration
        metrics: Current metrics
        step: Training step
        save_dir: Directory to save checkpoint
        filename: Checkpoint filename
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Prepare checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "metrics": metrics,
        "config": config
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    
    # Save checkpoint
    save_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, save_path)
    
    # Save config separately for easy access
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load training checkpoint
    
    Args:
        path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        scaler: Optional scaler to load state into
        map_location: Optional device mapping
        
    Returns:
        Dictionary with loaded metrics and config
    """
    checkpoint = torch.load(path, map_location=map_location)
    
    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
    # Load scaler state
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
    return {
        "step": checkpoint["step"],
        "metrics": checkpoint["metrics"],
        "config": checkpoint["config"]
    }

def get_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    distributed: bool = False
) -> DataLoader:
    """
    Create data loader with T4 optimizations
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        drop_last: Whether to drop last incomplete batch
        distributed: Whether to use DistributedSampler
        
    Returns:
        Configured DataLoader
    """
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None
        
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        sampler=sampler
    )
    
    return loader
