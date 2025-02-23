"""Memory tracking utilities for model training."""

import os
import psutil
import torch
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class MemoryTracker:
    """Tracks memory usage during model training."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics.
        
        Returns:
            dict: Dictionary containing memory statistics including:
                - system_memory_used: System memory usage in GB
                - cuda_memory_used: GPU memory usage in GB (if available)
                - tpu_memory_used: TPU memory usage in GB (if available)
        """
        stats = {
            'system_memory_used': self.process.memory_info().rss / (1024 * 1024 * 1024)
        }
        
        if torch.cuda.is_available():
            stats['cuda_memory_used'] = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
            stats['cuda_memory_cached'] = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)
            
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            if device.type == 'xla':
                stats['tpu_memory_used'] = xm.get_memory_info(device)['used'] / (1024 * 1024 * 1024)
        except ImportError:
            pass
            
        return stats
        
    def log_memory_stats(self, step: Optional[int] = None) -> None:
        """Log current memory statistics.
        
        Args:
            step (Optional[int]): Current training step for logging context
        """
        stats = self.get_memory_stats()
        prefix = f"Step {step}: " if step is not None else ""
        
        for name, value in stats.items():
            logger.info(f"{prefix}{name}: {value:.2f} GB")
            
    def get_peak_memory(self) -> Dict[str, float]:
        """Get peak memory usage statistics.
        
        Returns:
            dict: Dictionary containing peak memory statistics
        """
        stats = {}
        
        if torch.cuda.is_available():
            stats['peak_cuda_memory'] = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
            
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            if device.type == 'xla':
                stats['peak_tpu_memory'] = xm.get_memory_info(device)['peak'] / (1024 * 1024 * 1024)
        except ImportError:
            pass
            
        return stats
