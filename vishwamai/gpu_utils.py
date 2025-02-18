import torch
import math
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)

@dataclass
class GPUSpec:
    name: str
    memory_total: float
    memory_used: float
    compute_capability: Tuple[int, int]
    arch_list: list
    shared_memory: int
    max_threads_per_block: int
    max_thread_dims: Tuple[int, ...]

class GPUManager:
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. GPU required.")
        self.device = torch.device("cuda")
        self.properties = torch.cuda.get_device_properties(0)
        self._init_gpu_spec()
        
    def _init_gpu_spec(self) -> None:
        self.spec = GPUSpec(
            name=self.properties.name,
            memory_total=self.properties.total_memory / (1024**3),
            memory_used=torch.cuda.memory_allocated() / (1024**3),
            compute_capability=(self.properties.major, self.properties.minor),
            arch_list=self.properties.graphics_cap_list,
            shared_memory=self.properties.max_shared_memory_per_block,
            max_threads_per_block=self.properties.max_threads_per_block,
            max_thread_dims=self.properties.max_block_dim
        )
        
    def optimize_settings(self) -> None:
        """Apply optimal GPU settings."""
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        if self.spec.compute_capability >= (7, 0):
            # Enable tensor cores for Volta+
            torch.set_float32_matmul_precision('high')
            
        if hasattr(torch.cuda, 'memory_stats'):
            torch.cuda.empty_cache()
            # Use 95% of available memory
            torch.cuda.set_per_process_memory_fraction(0.95)
            
    def get_optimal_config(self) -> Dict:
        """Get optimal configuration based on GPU."""
        mem_available = self.spec.memory_total - self.spec.memory_used
        
        configs = {
            'T4': {
                'batch_size': min(2, max(1, int(mem_available * 0.3))),
                'seq_len': min(1024, max(128, int(mem_available * 256))),
                'dim_scale': 0.8,
                'layer_scale': 0.8
            },
            'V100': {
                'batch_size': min(4, max(1, int(mem_available * 0.4))),
                'seq_len': min(2048, max(256, int(mem_available * 512))),
                'dim_scale': 1.0,
                'layer_scale': 1.0
            },
            'A100': {
                'batch_size': min(8, max(2, int(mem_available * 0.5))),
                'seq_len': min(4096, max(512, int(mem_available * 1024))),
                'dim_scale': 1.2,
                'layer_scale': 1.2
            }
        }
        
        for gpu_type, config in configs.items():
            if gpu_type in self.spec.name:
                return config
                
        # Default conservative settings
        return {
            'batch_size': max(1, int(mem_available * 0.2)),
            'seq_len': min(512, max(64, int(mem_available * 128))),
            'dim_scale': 0.6,
            'layer_scale': 0.6
        }
        
    def adjust_model_config(self, model_args) -> None:
        """Adjust model configuration based on GPU capabilities."""
        config = self.get_optimal_config()
        
        # Apply basic configuration
        model_args.max_batch_size = config['batch_size']
        model_args.max_seq_len = config['seq_len']
        
        # Scale model dimensions
        model_args.dim = int(model_args.dim * config['dim_scale'])
        model_args.n_layers = int(model_args.n_layers * config['layer_scale'])
        
        # Adjust precision based on hardware
        if self.spec.compute_capability >= (8, 0):
            model_args.dtype = "bfloat16"
        elif self.spec.compute_capability >= (7, 0):
            model_args.dtype = "float16"
        else:
            model_args.dtype = "float32"
            
        logger.info(f"Adjusted configuration for {self.spec.name}:")
        logger.info(f"Batch size: {model_args.max_batch_size}")
        logger.info(f"Sequence length: {model_args.max_seq_len}")
        logger.info(f"Model dimension: {model_args.dim}")
        logger.info(f"Number of layers: {model_args.n_layers}")
        logger.info(f"Using dtype: {model_args.dtype}")
