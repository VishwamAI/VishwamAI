"""
DeepEP: Expert Parallelism with distributed processing via smallpond
"""

import torch
import smallpond
import numpy as np
from typing import Optional, Dict, Any, Tuple
import os

# Core DeepEP components
from .buffer import Buffer, get_buffer
from .config import Config, get_optimal_dispatch_config
from .utils import get_num_sms, set_num_sms, init_expert_parallel

class DistributedBuffer:
    """Distributed buffer management using smallpond"""
    
    def __init__(self, 
                num_executors: Optional[int] = None,
                cache_dir: Optional[str] = "/tmp/vishwamai/deepep_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize smallpond
        try:
            self.sp_session = smallpond.init(
                num_executors=num_executors or torch.cuda.device_count(),
                data_root=cache_dir,
                bind_numa_node=True
            )
        except:
            self.sp_session = None
            
    def distribute_computation(self,
                            x: torch.Tensor,
                            compute_fn: callable,
                            batch_size: Optional[int] = None) -> torch.Tensor:
        """Distribute tensor computation across executors"""
        if self.sp_session is None:
            return compute_fn(x)
            
        # Convert to numpy and create dataframe
        x_np = x.detach().cpu().numpy()
        df = self.sp_session.create_dataframe({'data': [x_np]})
        
        # Partition data
        if batch_size:
            df = df.repartition(
                self.sp_session.num_executors,
                batch_size=batch_size
            )
        else:
            df = df.repartition(self.sp_session.num_executors)
            
        # Process partitions
        def process_partition(partition):
            import torch
            import numpy as np
            data = torch.from_numpy(np.array(partition['data'].iloc[0]))
            result = compute_fn(data)
            return result.cpu().numpy()
            
        result_df = df.map_partitions(process_partition)
        
        # Gather results
        result = torch.from_numpy(result_df.to_pandas()['data'].iloc[0])
        return result.to(x.device)
        
    def cleanup(self):
        """Cleanup resources"""
        if self.sp_session:
            self.sp_session.shutdown()

# Enhanced Buffer with distributed processing
class EnhancedBuffer(Buffer):
    """Enhanced Buffer with smallpond integration"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distributed_buffer = DistributedBuffer()
        
    def dispatch_distributed(self, x, indices, weights=None):
        """Distributed dispatch operation"""
        def dispatch_fn(inputs):
            return super().dispatch(inputs, indices, weights)
            
        return self.distributed_buffer.distribute_computation(
            x, dispatch_fn
        )
        
    def combine_distributed(self, x, handle, weights=None):
        """Distributed combine operation"""
        def combine_fn(inputs):
            return super().combine(inputs, handle, weights)
            
        return self.distributed_buffer.distribute_computation(
            x, combine_fn
        )
        
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'distributed_buffer'):
            self.distributed_buffer.cleanup()

# Export components
__all__ = [
    'Buffer',
    'EnhancedBuffer',
    'Config',
    'DistributedBuffer',
    'get_buffer',
    'get_num_sms',
    'set_num_sms',
    'init_expert_parallel',
    'get_optimal_dispatch_config'
]

# Initialize on import
init_expert_parallel()
