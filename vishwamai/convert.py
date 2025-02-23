import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import numpy as np
import safetensors
from safetensors.torch import save_file, load_file
import torch
import jax.numpy as jnp
from dataclasses import asdict
from tqdm import tqdm
import datetime
from .model import ModelConfig, VishwamAIModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SafeModelConverter:
    """Handles secure model conversion using SafeTensors"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.metadata = {
            "framework_version": "1.0.0",
            "creation_date": "",
            "model_type": "VishwamAI"
        }

    def _validate_tensor(self, tensor: torch.Tensor, expected_shape: Optional[tuple] = None) -> bool:
        """Validate tensor properties for security"""
        if not isinstance(tensor, torch.Tensor):
            return False
        if expected_shape and tensor.shape != expected_shape:
            return False
        if not tensor.is_floating_point():  # Ensure floating point type
            return False
        return True

    def _secure_conversion(self, tensor: torch.Tensor) -> torch.Tensor:
        """Perform secure tensor conversion"""
        # Clone and detach for safety
        tensor = tensor.clone().detach()
        # Check for NaN/Inf values
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise ValueError("Tensor contains NaN or Inf values")
        return tensor

    def _convert_attention_weights(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Securely convert attention weights"""
        converted = {}
        attention_keys = [k for k in state_dict.keys() if 'attention' in k]
        
        for key in attention_keys:
            tensor = state_dict[key]
            # Validate shape based on config
            if 'query' in key or 'key' in key or 'value' in key:
                expected_shape = (self.config.heads, -1, self.config.head_dim)
                tensor = tensor.view(*expected_shape)
            converted[key] = self._secure_conversion(tensor)
            
        return converted

    def _convert_layer_norm(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert layer normalization weights"""
        converted = {}
        norm_keys = [k for k in state_dict.keys() if 'norm' in k or 'ln' in k]
        
        for key in norm_keys:
            tensor = state_dict[key]
            if 'weight' in key or 'bias' in key:
                # Validate shape
                if tensor.dim() != 1:
                    raise ValueError(f"Invalid LayerNorm tensor shape for {key}")
            converted[key] = self._secure_conversion(tensor)
            
        return converted

    def validate_checkpoint(self, checkpoint_path: str) -> bool:
        """Validate checkpoint integrity and security"""
        try:
            tensors = load_file(checkpoint_path)
            # Check metadata
            if not tensors.metadata:
                logger.warning("No metadata found in checkpoint")
                
            # Validate essential tensors
            required_keys = ['model.embed.weight', 'model.head.weight']
            for key in required_keys:
                if key not in tensors:
                    raise ValueError(f"Missing required tensor: {key}")
                    
            # Validate tensor shapes
            for name, tensor in tensors.items():
                if not self._validate_tensor(tensor):
                    raise ValueError(f"Invalid tensor: {name}")
                    
            return True
            
        except Exception as e:
            logger.error(f"Checkpoint validation failed: {str(e)}")
            return False

    def convert_to_safetensors(
        self, 
        input_path: str, 
        output_path: str, 
        source_format: str
    ) -> None:
        """Convert model weights to SafeTensors format"""
        logger.info(f"Converting {source_format} checkpoint to SafeTensors")
        
        # Load source weights
        if source_format == "pytorch":
            state_dict = torch.load(input_path, map_location='cpu')
        else:
            raise ValueError(f"Unsupported source format: {source_format}")

        # Convert weights with progress bar
        converted_weights = {}
        for key, tensor in tqdm(state_dict.items(), desc="Converting tensors"):
            # Secure conversion
            converted = self._secure_conversion(tensor)
            converted_weights[key] = converted

        # Add metadata
        self.metadata.update({
            "conversion_date": str(datetime.now()),
            "source_format": source_format,
            "model_config": json.dumps(asdict(self.config))
        })

        # Save in SafeTensors format
        save_file(converted_weights, output_path, metadata=self.metadata)
        logger.info(f"Successfully saved SafeTensors checkpoint to {output_path}")

    def merge_sharded_checkpoints(
        self, 
        shard_paths: List[str], 
        output_path: str
    ) -> None:
        """Merge sharded checkpoints into single SafeTensors file"""
        merged_weights = {}
        
        for shard_path in tqdm(shard_paths, desc="Merging shards"):
            shard = load_file(shard_path)
            # Validate shard
            if not self.validate_checkpoint(shard_path):
                raise ValueError(f"Invalid shard: {shard_path}")
            merged_weights.update(shard)

        # Save merged weights
        save_file(merged_weights, output_path, metadata=self.metadata)
        logger.info(f"Successfully merged shards to {output_path}")

def main():
    """CLI for safe model conversion"""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Securely convert models to SafeTensors format")
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--source-format", type=str, default="pytorch")
    parser.add_argument("--shard-paths", nargs="*", help="Paths to checkpoint shards")
    parser.add_argument("--merge-shards", action="store_true")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        with open(args.config) as f:
            config = ModelConfig(**json.load(f))
        
        converter = SafeModelConverter(config)
        
        if args.merge_shards and args.shard_paths:
            converter.merge_sharded_checkpoints(args.shard_paths, args.output_path)
        else:
            converter.convert_to_safetensors(
                args.input_path,
                args.output_path,
                args.source_format
            )
            
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
