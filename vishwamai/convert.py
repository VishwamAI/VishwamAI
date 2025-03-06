import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import numpy as np
import safetensors.flax as stf
from safetensors.flax import load_file, save_file
import torch
import jax
import jax.numpy as jnp
import datetime
from tqdm import tqdm
from .model import ModelConfig, VishwamAIModel
from google.cloud import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SafeModelConverter:
    """Handles secure model conversion using SafeTensors with TPU support"""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the SafeModelConverter with TPU support.

        Args:
            config (ModelConfig): The model configuration.
        """
        self.config = config
        self.metadata = {
            "framework_version": "1.0.0",
            "creation_date": datetime.datetime.utcnow().isoformat(),
            "model_type": "VishwamAI",
            "tpu_compatible": True,
            "jax_version": jax.__version__
        }
        # Initialize GCS client if TPU environment
        self.gcs_client = storage.Client() if os.getenv("CLOUD_TPU_WORKER_ID") else None

    def _get_file_from_gcs(self, path: str) -> str:
        """
        Download file from GCS if path is a GCS URI.

        Args:
            path (str): File path or GCS URI.

        Returns:
            str: Local file path.
        """
        if not path.startswith("gs://"):
            return path
            
        bucket_name, blob_name = path.replace("gs://", "").split("/", 1)
        bucket = self.gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        local_path = f"/tmp/{os.path.basename(path)}"
        blob.download_to_filename(local_path)
        return local_path

    def _save_to_gcs(self, local_path: str, gcs_path: str) -> None:
        """
        Upload file to GCS if path is a GCS URI.

        Args:
            local_path (str): Local file path.
            gcs_path (str): GCS URI destination.
        """
        if not gcs_path.startswith("gs://"):
            return
            
        bucket_name, blob_name = gcs_path.replace("gs://", "").split("/", 1)
        bucket = self.gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)

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
        if not tensor.is_floating_point():
            return False
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            return False
        return True

    def _secure_conversion(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform secure tensor conversion with TPU-compatible dtype.

        Args:
            tensor (torch.Tensor): The tensor to convert.

        Returns:
            torch.Tensor: The securely converted tensor.
        """
        # Convert to bfloat16 for TPU compatibility
        tensor = tensor.to(dtype=torch.bfloat16)
        
        # Clone and detach for safety
        tensor = tensor.clone().detach()
        
        # Replace NaN/Inf values with zeros/finite numbers
        tensor = torch.nan_to_num(tensor, nan=0.0, 
                                posinf=float(torch.finfo(tensor.dtype).max), 
                                neginf=float(torch.finfo(tensor.dtype).min))
        
        return tensor

    def _convert_to_jax(self, tensor: torch.Tensor) -> jnp.ndarray:
        """
        Convert PyTorch tensor to JAX array with TPU placement.

        Args:
            tensor (torch.Tensor): The PyTorch tensor to convert.

        Returns:
            jnp.ndarray: The converted JAX array on TPU.
        """
        numpy_array = tensor.detach().cpu().numpy()
        # Place on TPU if available, otherwise CPU
        device = jax.devices("tpu")[0] if jax.devices("tpu") else jax.devices("cpu")[0]
        return jax.device_put(numpy_array, device)

    def _convert_attention_weights(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Securely convert attention weights with TPU-optimized shapes.

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
                # Optimize shape for TPU
                head_dim = self.config.hidden_size // self.config.heads
                expected_shape = (self.config.heads, tensor.shape[-2], head_dim)
                tensor = tensor.view(*expected_shape)
            converted[key] = self._secure_conversion(tensor)
            
        return converted

    def _convert_layer_norm(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Same as before, already TPU-compatible"""
        converted = {}
        norm_keys = [k for k in state_dict.keys() if any(x in k for x in ['norm', 'ln'])]
        
        for key in norm_keys:
            tensor = state_dict[key]
            if 'weight' in key or 'bias' in key:
                if tensor.dim() != 1:
                    raise ValueError(f"Invalid LayerNorm tensor shape for {key}")
                if tensor.numel() != self.config.hidden_size:
                    raise ValueError(f"Invalid LayerNorm tensor size for {key}")
            converted[key] = self._secure_conversion(tensor)
            
        return converted

    def _validate_jax_tensor(self, tensor: jnp.ndarray, expected_shape: Optional[tuple] = None) -> bool:
        """Same as before"""
        if not isinstance(tensor, jnp.ndarray):
            return False
        if expected_shape and tensor.shape != expected_shape:
            return False
        if jnp.isnan(tensor).any() or jnp.isinf(tensor).any():
            return False
        return True

    def validate_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Validate checkpoint integrity with TPU compatibility checks.

        Args:
            checkpoint_path (str): The path to the checkpoint file.

        Returns:
            bool: True if the checkpoint is valid, False otherwise.
        """
        try:
            # Handle GCS paths
            local_path = self._get_file_from_gcs(checkpoint_path)
            tensors = load_file(local_path)
            metadata = tensors.metadata or {}
            
            # Validate metadata fields
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
        Convert model weights to SafeTensors format with TPU optimizations.

        Args:
            input_path (str): The path to the input checkpoint file.
            output_path (str): The path to save the converted SafeTensors file.
            source_format (str): The format of the source checkpoint (e.g., "pytorch").
        """
        logger.info(f"Converting {source_format} checkpoint to TPU-compatible SafeTensors")
        
        # Handle GCS paths for input
        local_input = self._get_file_from_gcs(input_path)
        
        # Load source weights
        if source_format == "pytorch":
            state_dict = torch.load(local_input, map_location='cpu')
        else:
            raise ValueError(f"Unsupported source format: {source_format}")

        # Convert weights with progress bar
        jax_weights = {}
        for key, tensor in tqdm(state_dict.items(), desc="Converting tensors"):
            # Secure conversion and transform to JAX
            tensor = self._secure_conversion(tensor)  # Now uses bfloat16
            jax_array = self._convert_to_jax(tensor)  # Now places on TPU
            jax_weights[key] = jax_array

        # Update metadata for TPU compatibility
        self.metadata.update({
            "conversion_date": datetime.datetime.utcnow().isoformat(),
            "source_format": source_format,
            "target_format": "jax",
            "model_config": json.dumps(self.config.__dict__),
            "tpu_sharded": jax.device_count() > 1,
            "num_devices": jax.device_count()
        })

        # Save locally first
        local_output = output_path if not output_path.startswith("gs://") else f"/tmp/{os.path.basename(output_path)}"
        save_file(jax_weights, local_output, metadata=self.metadata)

        # Upload to GCS if needed
        if output_path.startswith("gs://"):
            self._save_to_gcs(local_output, output_path)
            os.remove(local_output)  # Cleanup

        logger.info(f"Successfully saved TPU-compatible JAX SafeTensors checkpoint to {output_path}")

    def merge_sharded_checkpoints(
        self, 
        shard_paths: List[str], 
        output_path: str
    ) -> None:
        """
        Merge sharded checkpoints with TPU optimizations.

        Args:
            shard_paths (List[str]): List of paths to the checkpoint shards.
            output_path (str): The path to save the merged SafeTensors file.
        """
        merged_weights = {}
        metadata = {}
        
        for shard_path in tqdm(shard_paths, desc="Merging shards"):
            # Handle GCS paths
            local_path = self._get_file_from_gcs(shard_path)
            
            # Load and validate shard
            if not self.validate_checkpoint(local_path):
                raise ValueError(f"Invalid shard: {shard_path}")
                
            shard = load_file(local_path)
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
            "target_format": "jax",
            "tpu_compatible": True,
            "tpu_sharded": jax.device_count() > 1
        })

        # Save locally first
        local_output = output_path if not output_path.startswith("gs://") else f"/tmp/{os.path.basename(output_path)}"
        save_file(merged_weights, local_output, metadata=metadata)

        # Upload to GCS if needed
        if output_path.startswith("gs://"):
            self._save_to_gcs(local_output, output_path)
            os.remove(local_output)  # Cleanup

        logger.info(f"Successfully merged {len(shard_paths)} shards to {output_path}")

def main():
    """CLI for safe model conversion with TPU support"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Securely convert models to TPU-compatible SafeTensors format")
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--source-format", type=str, default="pytorch")
    parser.add_argument("--shard-paths", nargs="*", help="Paths to checkpoint shards")
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--use-tpu", action="store_true", help="Enable TPU-specific optimizations")
    
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
