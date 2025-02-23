"""TPU and XLA utilities for distributed training."""
from typing import Any, Dict, Optional, Union, List, Tuple
import os
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.utils.data import DataLoader, Dataset

def initialize_tpu(use_tpu: bool = True) -> Optional[torch.device]:
    """Initialize TPU device and environment.
    
    Args:
        use_tpu: Whether to use TPU
        
    Returns:
        TPU device if available, else None
    """
    if not use_tpu:
        return None
        
    if xm.xrt_world_size() < 1:
        raise RuntimeError("No TPU devices found")
        
    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()
    
    # Set up environment variables
    os.environ['XLA_USE_BF16'] = '1'  # Enable bfloat16
    os.environ['TPU_NUM_DEVICES'] = str(world_size)
    
    return device

def create_xla_model(
    model: torch.nn.Module,
    device: Optional[torch.device] = None
) -> torch.nn.Module:
    """Move model to XLA device and prepare for training.
    
    Args:
        model: PyTorch model
        device: Optional XLA device
        
    Returns:
        Model prepared for XLA
    """
    if device is None:
        device = xm.xla_device()
        
    # Move model to device
    model = model.to(device)
    
    # Enable grad scaling for mixed precision
    if hasattr(model, 'half'):
        try:
            model = model.half()
        except:
            pass
            
    return model

def move_to_device(
    data: Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]],
    device: Optional[torch.device] = None
) -> Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]]:
    """Move data to XLA device.
    
    Args:
        data: Input tensors/dict/list
        device: Optional XLA device
        
    Returns:
        Data moved to device
    """
    if device is None:
        device = xm.xla_device()
        
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(x, device) for x in data]
    else:
        return data

def xla_data_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    device: Optional[torch.device] = None,
    mp_device_loader: bool = True
) -> Union[DataLoader, pl.MpDeviceLoader]:
    """Create data loader optimized for XLA.
    
    Args:
        dataset: Input dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        drop_last: Whether to drop last incomplete batch
        device: Optional XLA device
        mp_device_loader: Whether to use MpDeviceLoader
        
    Returns:
        XLA-optimized data loader
    """
    if device is None:
        device = xm.xla_device()
        
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True
    )
    
    if mp_device_loader:
        loader = pl.MpDeviceLoader(loader, device)
        
    return loader

def mark_step():
    """Mark step for XLA execution."""
    xm.mark_step()

def wait_device_ops():
    """Wait for all device operations to complete."""
    xm.wait_device_ops()

def save_to_xla(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: str,
    **kwargs
):
    """Save checkpoint compatible with XLA.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        filepath: Path to save checkpoint
        **kwargs: Additional items to save
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **kwargs
    }
    xm.save(save_dict, filepath)
    
def load_from_xla(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: str
) -> Dict[str, Any]:
    """Load XLA-compatible checkpoint.
    
    Args:
        model: Model to load into
        optimizer: Optimizer to load into
        filepath: Path to checkpoint
        
    Returns:
        Dictionary with additional saved items
    """
    checkpoint = xm.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Remove standard items and return rest
    del checkpoint['model_state_dict']
    del checkpoint['optimizer_state_dict']
    return checkpoint

def xla_spawn(
    fn: callable,
    args: Tuple = (),
    nprocs: Optional[int] = None,
    start_method: str = 'spawn'
):
    """Spawn processes for TPU training.
    
    Args:
        fn: Function to run in processes
        args: Arguments to pass to function
        nprocs: Number of processes (default: number of TPU cores)
        start_method: Process start method
    """
    if nprocs is None:
        nprocs = xm.xrt_world_size()
        
    return xmp.spawn(fn, args=args, nprocs=nprocs, start_method=start_method)
