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
        """
        Initialize the SafeModelConverter with the provided model configuration.
        
        This constructor sets up the converter with the specified model parameters and
        initializes crucial metadata for the conversion process, including framework
        version, creation date, and model type.
        
        Args:
            config (ModelConfig): The configuration object with model parameters and conversion settings.
        """
        self.config = config
        self.metadata = {
            "framework_version": "1.0.0",
            "creation_date": "",
            "model_type": "VishwamAI"
        }

    def _validate_tensor(self, tensor: torch.Tensor, expected_shape: Optional[tuple] = None) -> bool:
        """
        Check if a tensor is a valid floating-point tensor and matches an optional shape.
        
        This function verifies that the input is a torch.Tensor, that it matches the expected shape
        (if provided), and that it is of a floating-point type.
        
        Args:
            tensor (torch.Tensor): The tensor to validate.
            expected_shape (Optional[tuple]): The required shape for the tensor, if applicable.
        
        Returns:
            bool: True if the tensor meets all criteria; otherwise, False.
        """
        if not isinstance(tensor, torch.Tensor):
            return False
        if expected_shape and tensor.shape != expected_shape:
            return False
        if not tensor.is_floating_point():  # Ensure floating point type
            return False
        return True

    def _secure_conversion(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Securely clones and validates the input tensor.
        
        This method creates a detached clone of the given tensor and checks it for any invalid 
        numerical values (NaN or Inf). A ValueError is raised if any such values are found.
        
        Args:
            tensor (torch.Tensor): The tensor to secure.
        
        Returns:
            torch.Tensor: A detached clone of the tensor verified for numerical integrity.
        
        Raises:
            ValueError: If the tensor contains NaN or Inf values.
        """
        # Clone and detach for safety
        tensor = tensor.clone().detach()
        # Check for NaN/Inf values
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise ValueError("Tensor contains NaN or Inf values")
        return tensor

    def _convert_attention_weights(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts attention weights in a model's state dictionary securely.
        
        This function filters the given state dictionary to identify keys associated with
        attention layers. For tensors whose keys contain 'query', 'key', or 'value', it
        reshapes them to the expected dimensions using the model configuration (i.e.,
        (heads, -1, head_dim)). Each tensor is then processed via a secure conversion 
        routine that clones and detaches it to ensure integrity.
        
        Args:
            state_dict (Dict[str, torch.Tensor]): Mapping of model weight names to tensors.
        
        Returns:
            Dict[str, torch.Tensor]: A dictionary of the securely converted attention weights.
        """
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
        """
        Converts layer normalization weights from the model state dictionary.
        
        This function filters the state dictionary to extract keys related to layer
        normalization (identified by 'norm' or 'ln' in their names). For each key,
        if it pertains to 'weight' or 'bias', the associated tensor is verified to be
        one-dimensional before it is securely converted via detachment and cloning.
        A ValueError is raised if a 'weight' or 'bias' tensor does not have a one-dimensional shape.
        
        Args:
            state_dict (Dict[str, torch.Tensor]): A dictionary mapping model weight names
                to their corresponding tensors.
        
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the securely converted
                layer normalization weights.
        
        Raises:
            ValueError: If a tensor for 'weight' or 'bias' does not have one dimension.
        """
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
        """
        Validate the integrity and correctness of a checkpoint file.
        
        This function loads a checkpoint from the specified path and checks that it contains the required metadata and essential model weight tensors. It verifies that each tensor meets the expected properties and shape. If any required tensor is missing or fails validation, the function logs an error and returns False.
        
        Args:
            checkpoint_path: The file path to the checkpoint.
        
        Returns:
            True if the checkpoint passes all validation checks; False otherwise.
        """
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
        """
        Convert model weights from a supported source format to SafeTensors format.
        
        Loads a PyTorch checkpoint, applies a secure conversion to each tensor, updates
        conversion metadata with the conversion date, source format, and model configuration,
        and saves the converted weights as a SafeTensors file.
        
        Args:
            input_path: The file path to the input checkpoint.
            output_path: The file path to save the resulting SafeTensors file.
            source_format: The format of the source checkpoint; only "pytorch" is supported.
        
        Raises:
            ValueError: If the specified source_format is not supported.
        """
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
        """
        Merge multiple sharded checkpoint files into a single SafeTensors file.
        
        Loads and validates each shard before merging their weights. The merged weights,
        along with conversion metadata, are saved to the specified output path. A ValueError
        is raised if any shard fails validation.
        
        Args:
            shard_paths (List[str]): Paths to individual checkpoint shard files.
            output_path (str): Destination path for the merged SafeTensors file.
        
        Raises:
            ValueError: If any checkpoint shard is invalid.
        """
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
