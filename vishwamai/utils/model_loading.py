"""Utilities for loading model weights from different formats."""

import jax
import jax.numpy as jnp
from safetensors.torch import load_file
import numpy as np
from typing import Dict, Any, Optional
from huggingface_hub import hf_hub_download
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

def download_gemma_weights(
    model_id: str = "google/gemma-3-27b-pt",
    cache_dir: Optional[str] = None
) -> Dict[str, Path]:
    """Download Gemma weights from Hugging Face Hub."""
    files = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "config.json",
        "tokenizer.model"
    ]
    
    downloaded_files = {}
    for file in files:
        try:
            path = hf_hub_download(
                repo_id=model_id,
                filename=file,
                cache_dir=cache_dir
            )
            downloaded_files[file] = Path(path)
            logger.info(f"Downloaded {file} to {path}")
        except Exception as e:
            logger.error(f"Failed to download {file}: {e}")
            raise
            
    return downloaded_files

def convert_gemma_safetensors_to_jax(
    safetensors_paths: list[Path],
    dtype: jnp.dtype = jnp.bfloat16
) -> Dict[str, Any]:
    """Convert Gemma safetensors weights to JAX format."""
    jax_weights = {}
    
    # Load and merge all safetensors files
    for path in safetensors_paths:
        tensors = load_file(path)
        for key, tensor in tensors.items():
            # Convert PyTorch tensor to numpy then JAX array
            numpy_tensor = tensor.cpu().numpy()
            
            # Handle parameter name mapping
            jax_key = _convert_gemma_param_name(key)
            
            # Convert to specified dtype
            jax_array = jnp.array(numpy_tensor, dtype=dtype)
            
            jax_weights[jax_key] = jax_array
            
    return jax_weights

def _convert_gemma_param_name(pt_name: str) -> str:
    """Convert Gemma parameter names from PyTorch to JAX/Flax format."""
    # Example mappings (expand based on actual model structure):
    name_map = {
        "transformer.": "",
        "attention.query": "attention.q_proj",
        "attention.key": "attention.k_proj",
        "attention.value": "attention.v_proj",
        "attention.dense": "attention.o_proj",
        "input_layernorm": "ln_1",
        "post_attention_layernorm": "ln_2",
        "mlp.dense_h_to_4h": "mlp.dense1",
        "mlp.dense_4h_to_h": "mlp.dense2",
    }
    
    mapped_name = pt_name
    for pt_pattern, jax_pattern in name_map.items():
        mapped_name = mapped_name.replace(pt_pattern, jax_pattern)
        
    return mapped_name

def load_pretrained_weights(
    model_id: str,
    cache_dir: Optional[str] = None,
    dtype: jnp.dtype = jnp.bfloat16
) -> Dict[str, Any]:
    """Main function to load pretrained Gemma weights."""
    # Download weights
    downloaded_files = download_gemma_weights(model_id, cache_dir)
    
    # Get safetensors files
    safetensors_files = [
        path for name, path in downloaded_files.items()
        if name.endswith('.safetensors')
    ]
    
    # Convert to JAX format
    jax_weights = convert_gemma_safetensors_to_jax(
        safetensors_files,
        dtype=dtype
    )
    
    return jax_weights