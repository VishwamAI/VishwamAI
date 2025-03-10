import os
import json
from argparse import ArgumentParser
from glob import glob
from typing import Dict, Any
import jax
import jax.numpy as jnp
from flax import serialization
from flax.training import checkpoints
import numpy as np
from tqdm import tqdm
from safetensors import safe_open
from safetensors.flax import save_file

def weight_dequant(weight: jnp.ndarray, scale_inv: jnp.ndarray) -> jnp.ndarray:
    """
    Dequantizes FP8 weights using scale inverse.
    
    Args:
        weight (jnp.ndarray): The FP8 weight tensor stored as uint8
        scale_inv (jnp.ndarray): The inverse scale tensor
        
    Returns:
        jnp.ndarray: The dequantized weight tensor in bfloat16
    """
    # Convert uint8 (FP8) to float32 first, then apply scale
    weight_float = weight.astype(jnp.float32)
    # Apply the inverse scale (element-wise division)
    dequantized = weight_float / scale_inv
    # Convert to bfloat16
    return dequantized.astype(jnp.bfloat16)

def main(fp8_path, bf16_path):
    """
    Converts FP8 weights to BF16 and saves the converted weights.

    This function reads FP8 weights from the specified directory, converts them to BF16,
    and saves the converted weights to another specified directory. It also updates the
    model index file to reflect the changes.

    Args:
    fp8_path (str): The path to the directory containing the FP8 weights and model index file.
    bf16_path (str): The path to the directory where the converted BF16 weights will be saved.

    Raises:
    KeyError: If a required scale_inv tensor is missing for a weight.

    Notes:
    - The function assumes that the FP8 weights are stored in safetensor files.
    - The function caches loaded safetensor files to optimize memory usage.
    - The function updates the model index file to remove references to scale_inv tensors.
    """
    os.makedirs(bf16_path, exist_ok=True)
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    
    weight_map = model_index["weight_map"]
    
    # Cache for loaded safetensor files
    loaded_files = {}
    fp8_weight_names = []

    # Helper function to get tensor from the correct file
    def get_tensor(tensor_name):
        """
        Retrieves a tensor from the cached safetensor files or loads it from disk if not cached.

        Args:
            tensor_name (str): The name of the tensor to retrieve.

        Returns:
            jnp.ndarray: The retrieved tensor.

        Raises:
            KeyError: If the tensor does not exist in the safetensor file.
        """
        file_name = weight_map[tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(fp8_path, file_name)
            # Open the safetensor file in read-only mode
            loaded_files[file_name] = {}
            with safe_open(file_path, framework="flax") as f:
                for key in f.keys():
                    # Load tensor to device
                    loaded_files[file_name][key] = jax.device_put(f.get_tensor(key))
        
        return loaded_files[file_name][tensor_name]

    # Get all safetensor files in the directory
    safetensor_files = list(glob(os.path.join(fp8_path, "*.safetensors")))
    safetensor_files.sort()
    
    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        
        # Load the current safetensor file
        current_state_dict = {}
        with safe_open(safetensor_file, framework="flax") as f:
            for key in f.keys():
                current_state_dict[key] = jax.device_put(f.get_tensor(key))
        
        loaded_files[file_name] = current_state_dict
        
        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            if weight_name.endswith("_scale_inv"):
                continue
            # Check if this is an FP8 weight (element size 1 byte)
            elif weight.dtype == jnp.uint8:  # FP8 weight
                scale_inv_name = f"{weight_name}_scale_inv"
                try:
                    # Get scale_inv from the correct file
                    scale_inv = get_tensor(scale_inv_name)
                    fp8_weight_names.append(weight_name)
                    new_state_dict[weight_name] = weight_dequant(weight, scale_inv)
                except KeyError:
                    print(f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion")
                    new_state_dict[weight_name] = weight
            else:
                new_state_dict[weight_name] = weight
        
        # Save the converted state dict
        new_safetensor_file = os.path.join(bf16_path, file_name)
        save_file(new_state_dict, new_safetensor_file)
        
        # Memory management: keep only the 2 most recently used files
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            jax.clear_caches()  # Clear JAX caches
    
    # Update model index
    new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    for weight_name in fp8_weight_names:
        scale_inv_name = f"{weight_name}_scale_inv"
        if scale_inv_name in weight_map:
            weight_map.pop(scale_inv_name)
    
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp8-hf-path", type=str, required=True,
                        help="Path to the directory containing FP8 weights")
    parser.add_argument("--output-bf16-hf-path", type=str, required=True,
                        help="Path where converted BF16 weights will be saved")
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_bf16_hf_path)