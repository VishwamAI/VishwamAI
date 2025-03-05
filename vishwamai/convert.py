import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import numpy as np
import safetensors.flax as stf
from safetensors.flax import load_file, save_file
import torch
import jax.numpy as jnp
import datetime
from tqdm import tqdm
from .model import ModelConfig, VishwamAIModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SafeModelConverter:
    """Handles secure model conversion using SafeTensors"""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the SafeModelConverter with the given configuration.

        Args:
            config (ModelConfig): The model configuration.
        """
        self.config = config
        self.metadata = {
            "framework_version": "1.0.0",
            "creation_date": datetime.datetime.utcnow().isoformat(),
            "model_type": "VishwamAI"
        }

    def _validate_tensor(self, tensor: torch.Tensor, expected_shape: Optional[tuple] = None) -> bool:
        """
        Validate tensor properties for security.

        Args:
            tensor (torch.Tensor): The tensor to validate.
            expected_shape (Optional[tuple]): The expected shape of the tensor.

        Returns:
            bool: True if the tensor is valid, False otherwise.
        """
        if not isinstance(tensor, torch.Tensor):
            return False
        if expected_shape and tensor.shape != expected_shape:
            return False
        if not tensor.is_floating_point():  # Ensure floating point type
            return False
        # Check for NaN or Inf values
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            return False
        return True

    def _secure_conversion(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform secure tensor conversion with automatic dtype handling.

        Args:
            tensor (torch.Tensor): The tensor to convert.

        Returns:
            torch.Tensor: The securely converted tensor.
        """
        # Convert to float32 for consistency
        tensor = tensor.to(dtype=torch.float32)
        
        # Clone and detach for safety
        tensor = tensor.clone().detach()
        
        # Replace NaN/Inf values with zeros/finite numbers
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=float(torch.finfo(tensor.dtype).max), 
                                neginf=float(torch.finfo(tensor.dtype).min))
        
        return tensor

    def _convert_to_jax(self, tensor: torch.Tensor) -> jnp.ndarray:
        """
        Convert PyTorch tensor to JAX array.

        Args:
            tensor (torch.Tensor): The PyTorch tensor to convert.

        Returns:
            jnp.ndarray: The converted JAX array.
        """
        # Convert to numpy first, then to JAX
        return jnp.array(tensor.detach().cpu().numpy())

    def _convert_attention_weights(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Securely convert attention weights with dynamic shape handling.

        Args:
            state_dict (Dict[str, torch.Tensor]): The state dictionary containing model weights.

        Returns:
            Dict[str, torch.Tensor]: The converted attention weights.
        """
        converted = {}
        attention_keys = [k for k in state_dict.keys() if 'attention' in k]
        
        for key in attention_keys:
            tensor = state_dict[key]
            if 'query' in key or 'key' in key or 'value' in key:
                # Use dynamic shape based on input tensor
                expected_shape = (self.config.heads, tensor.shape[-2], self.config.head_dim)
                tensor = tensor.view(*expected_shape)
            converted[key] = self._secure_conversion(tensor)
            
        return converted

    def _convert_layer_norm(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Convert layer normalization weights with additional validation.

        Args:
            state_dict (Dict[str, torch.Tensor]): The state dictionary containing model weights.

        Returns:
            Dict[str, torch.Tensor]: The converted layer normalization weights.
        """
        converted = {}
        norm_keys = [k for k in state_dict.keys() if any(x in k for x in ['norm', 'ln'])]
        
        for key in norm_keys:
            tensor = state_dict[key]
            if 'weight' in key or 'bias' in key:
                if tensor.dim() != 1:
                    raise ValueError(f"Invalid LayerNorm tensor shape for {key}: expected 1D, got {tensor.dim()}D")
                if tensor.numel() != self.config.hidden_size:
                    raise ValueError(f"Invalid LayerNorm tensor size for {key}: expected {self.config.hidden_size}, got {tensor.numel()}")
            converted[key] = self._secure_conversion(tensor)
            
        return converted

    def _validate_jax_tensor(self, tensor: jnp.ndarray, expected_shape: Optional[tuple] = None) -> bool:
        """
        Validate JAX array properties for security.

        Args:
            tensor (jnp.ndarray): The JAX array to validate.
            expected_shape (Optional[tuple]): The expected shape of the array.

        Returns:
            bool: True if the array is valid, False otherwise.
        """
        if not isinstance(tensor, jnp.ndarray):
            return False
        if expected_shape and tensor.shape != expected_shape:
            return False
        # Check for NaN or Inf values
        if jnp.isnan(tensor).any() or jnp.isinf(tensor).any():
            return False
        return True

    def validate_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Validate checkpoint integrity and security with enhanced metadata handling.

        Args:
            checkpoint_path (str): The path to the checkpoint file.

        Returns:
            bool: True if the checkpoint is valid, False otherwise.
        """
        try:
            tensors = load_file(checkpoint_path)
            # Safely handle metadata
            metadata = tensors.metadata or {}
            
            # Validate metadata fields if present
            if metadata:
                required_metadata = ['framework_version', 'model_type']
                for field in required_metadata:
                    if field not in metadata:
                        logger.warning(f"Missing metadata field: {field}")
                
            # Validate essential tensors
            required_keys = ['model.embed.weight', 'model.head.weight']
            for key in required_keys:
                if key not in tensors:
                    raise ValueError(f"Missing required tensor: {key}")
                    
            # Validate tensor shapes and values
            for name, tensor in tensors.items():
                if not self._validate_jax_tensor(tensor):
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
        """
        Convert model weights to SafeTensors format with enhanced metadata.

        Args:
            input_path (str): The path to the input checkpoint file.
            output_path (str): The path to save the converted SafeTensors file.
            source_format (str): The format of the source checkpoint (e.g., "pytorch").
        """
        logger.info(f"Converting {source_format} checkpoint to SafeTensors")
        
        # Load source weights
        if source_format == "pytorch":
            state_dict = torch.load(input_path, map_location='cpu')
        else:
            raise ValueError(f"Unsupported source format: {source_format}")

        # Convert weights with progress bar
        jax_weights = {}
        for key, tensor in tqdm(state_dict.items(), desc="Converting tensors"):
            # Secure conversion and transform to JAX
            tensor = self._secure_conversion(tensor)
            jax_array = self._convert_to_jax(tensor)
            jax_weights[key] = jax_array

        # Update metadata with JSON-safe values
        self.metadata.update({
            "conversion_date": datetime.datetime.utcnow().isoformat(),
            "source_format": source_format,
            "target_format": "jax",
            "model_config": json.dumps(self.config.__dict__)
        })

        # Save in SafeTensors format using JAX API
        save_file(jax_weights, output_path, metadata=self.metadata)
        logger.info(f"Successfully saved JAX SafeTensors checkpoint to {output_path}")

    def merge_sharded_checkpoints(
        self, 
        shard_paths: List[str], 
        output_path: str
    ) -> None:
        """
        Merge sharded checkpoints into single SafeTensors file with validation.

        Args:
            shard_paths (List[str]): List of paths to the checkpoint shards.
            output_path (str): The path to save the merged SafeTensors file.
        """
        merged_weights = {}
        metadata = {}
        
        for shard_path in tqdm(shard_paths, desc="Merging shards"):
            # Load and validate shard
            if not self.validate_checkpoint(shard_path):
                raise ValueError(f"Invalid shard: {shard_path}")
                
            shard = load_file(shard_path)
            shard_metadata = shard.metadata or {}
            
            # Update metadata from shards
            metadata.update(shard_metadata)
            
            # JAX arrays are directly added to merged weights
            for key, value in shard.items():
                merged_weights[key] = value

        # Add merge-specific metadata
        metadata.update({
            "merge_date": datetime.datetime.utcnow().isoformat(),
            "num_shards": len(shard_paths),
            "target_format": "jax"
        })

        # Save merged weights using JAX API
        save_file(merged_weights, output_path, metadata=metadata)
        logger.info(f"Successfully merged {len(shard_paths)} shards to {output_path}")

def main():
    """CLI for safe model conversion"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Securely convert models to SafeTensors format")
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--source-format", type=str, default="pytorch")
    parser.add_argument("--shard-paths", nargs="*", help="Paths to checkpoint shards")
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--to-jax", action="store_true", help="Convert to JAX format")
    
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
