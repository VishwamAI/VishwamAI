import torch
from safetensors.torch import save_file, load_file
from typing import Dict, Any, Optional
import numpy as np
import zlib
import os

class CheckpointManager:
    """Handles efficient model checkpoint saving and loading."""
    
    def __init__(self, compression: bool = True, shard_size: int = 1024*1024*1024):
        self.compression = compression
        self.shard_size = shard_size  # 1GB default shard size
        
    def quantize_weights(self, tensor: torch.Tensor, bits: int = 8) -> Dict:
        """Quantize tensor weights for storage."""
        if bits not in [8, 4, 2]:
            raise ValueError("Only 8, 4, or 2 bit quantization supported")
            
        scale = (tensor.abs().max() / (2**(bits-1)-1)).float()
        quantized = (tensor / scale).round().char()
        
        return {
            'quantized': quantized,
            'scale': scale,
            'bits': bits
        }
        
    def dequantize_weights(self, data: Dict) -> torch.Tensor:
        """Dequantize tensor weights."""
        return data['quantized'].float() * data['scale']
        
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        filepath: str,
        extra_data: Optional[Dict[str, Any]] = None,
        quantize: bool = True
    ):
        """Save checkpoint with compression and sharding."""
        # Get state dictionaries
        model_state = model.state_dict()
        optimizer_state = optimizer.state_dict() if optimizer else None
        
        # Quantize if enabled
        if quantize:
            quantized_state = {}
            for name, tensor in model_state.items():
                if tensor.dtype in [torch.float32, torch.float16]:
                    quantized_state[name] = self.quantize_weights(tensor)
                else:
                    quantized_state[name] = tensor
            model_state = quantized_state
            
        # Prepare checkpoint data
        checkpoint = {
            'model_state': model_state,
            'optimizer_state': optimizer_state,
            'extra_data': extra_data
        }
        
        # Split into shards if needed
        if os.path.getsize(filepath) > self.shard_size:
            self._save_sharded(checkpoint, filepath)
        else:
            # Save as safetensors format
            save_file(checkpoint, filepath + '.safetensors')
            
        # Save metadata
        metadata = {
            'format_version': '1.0',
            'is_sharded': os.path.getsize(filepath) > self.shard_size,
            'quantization': quantize,
            'compression': self.compression
        }
        torch.save(metadata, filepath + '.meta')
        
    def _save_sharded(self, checkpoint: Dict[str, Any], filepath: str):
        """Save large checkpoints as shards."""
        shard_idx = 0
        current_shard = {}
        current_size = 0
        
        for key, value in checkpoint.items():
            value_size = value.element_size() * value.nelement()
            
            if current_size + value_size > self.shard_size:
                # Save current shard
                shard_path = f"{filepath}.shard{shard_idx}.safetensors"
                save_file(current_shard, shard_path)
                
                # Start new shard
                shard_idx += 1
                current_shard = {key: value}
                current_size = value_size
            else:
                current_shard[key] = value
                current_size += value_size
                
        # Save final shard
        if current_shard:
            shard_path = f"{filepath}.shard{shard_idx}.safetensors"
            save_file(current_shard, shard_path)
            
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        filepath: str
    ) -> Dict[str, Any]:
        """Load checkpoint with automatic format detection."""
        # Load metadata
        metadata = torch.load(filepath + '.meta')
        
        if metadata['is_sharded']:
            checkpoint = self._load_sharded(filepath)
        else:
            checkpoint = load_file(filepath + '.safetensors')
            
        # Dequantize if needed
        if metadata['quantization']:
            model_state = {}
            for name, data in checkpoint['model_state'].items():
                if isinstance(data, dict) and 'quantized' in data:
                    model_state[name] = self.dequantize_weights(data)
                else:
                    model_state[name] = data
            checkpoint['model_state'] = model_state
            
        # Load states
        model.load_state_dict(checkpoint['model_state'])
        if optimizer and checkpoint['optimizer_state']:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            
        return checkpoint.get('extra_data', {})
        
    def _load_sharded(self, filepath: str) -> Dict[str, Any]:
        """Load sharded checkpoint."""
        checkpoint = {}
        shard_idx = 0
        
        while True:
            shard_path = f"{filepath}.shard{shard_idx}.safetensors"
            if not os.path.exists(shard_path):
                break
                
            shard_data = load_file(shard_path)
            checkpoint.update(shard_data)
            shard_idx += 1
            
        return checkpoint
