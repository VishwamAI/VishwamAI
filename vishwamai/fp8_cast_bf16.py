import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm
import torch
import warnings
from safetensors.torch import load_file, save_file

from .kernel import (
    act_quant,
    weight_dequant,
    optimize_kernel_layout,
    prepare_kernel,
    fuse_kernels
)

def setup_model_precision(model: torch.nn.Module, precision: str = "auto", save_path: str = None) -> torch.nn.Module:
    """Setup model precision based on hardware capabilities."""
    try:
        if precision == "bf16":
            if torch.cuda.is_bf16_supported():
                model = model.to(dtype=torch.bfloat16)
            else:
                warnings.warn("BF16 not supported, falling back to FP32")
                model = model.to(dtype=torch.float32)
        elif precision == "fp16":
            if torch.cuda.is_available():
                model = model.to(dtype=torch.float16)
            else:
                warnings.warn("FP16 not supported, falling back to FP32")
                model = model.to(dtype=torch.float32)
        elif precision == "fp8":
            if hasattr(torch, 'float8_e4m3fn'):
                # Apply kernel optimizations for FP8
                for name, param in model.named_parameters():
                    if param.dim() >= 2:  # Only process matrices
                        param.data = optimize_kernel_layout(param.data)
                        if save_path:
                            prepared_weight, quant_params = prepare_kernel(param.data, quantize=True)
                            param.data = prepared_weight
                model = model.to(dtype=torch.float8_e4m3fn)
            else:
                warnings.warn("FP8 not supported, falling back to BF16/FP16")
                return setup_model_precision(model, "auto", save_path)
        elif precision == "auto":
            if torch.cuda.is_bf16_supported():
                model = model.to(dtype=torch.bfloat16)
            elif torch.cuda.is_available():
                model = model.to(dtype=torch.float16)
            else:
                model = model.to(dtype=torch.float32)
        else:
            warnings.warn(f"Unknown precision {precision}, using FP32")
            model = model.to(dtype=torch.float32)
    except Exception as e:
        warnings.warn(f"Error setting precision: {str(e)}, falling back to FP32")
        model = model.to(dtype=torch.float32)
    
    return model

def get_optimal_precision(hardware_info: dict = None) -> str:
    """Determine optimal precision based on hardware info."""
    try:
        if hasattr(torch, 'float8_e4m3fn'):
            return "fp8"
        if torch.cuda.is_bf16_supported():
            return "bf16"
        elif torch.cuda.is_available():
            return "fp16"
    except Exception as e:
        warnings.warn(f"Error determining optimal precision: {str(e)}")
    return "fp32"

def load_and_process_weights(input_path: str, output_path: str = None):
    """Load weights and process them with kernel optimizations."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path {input_path} does not exist")
        
    if output_path:
        os.makedirs(output_path, exist_ok=True)
    
    weights = {}
    try:
        weights = load_file(input_path)
        
        # Apply kernel optimizations
        for name, weight in weights.items():
            if weight.dim() >= 2:  # Only process matrices
                weight = optimize_kernel_layout(weight)
                if hasattr(torch, 'float8_e4m3fn'):
                    # Quantize to FP8 if supported
                    weight_quant, _ = prepare_kernel(weight, quantize=True)
                    weights[name] = weight_quant
                else:
                    weights[name] = weight
                    
        if output_path:
            save_file(weights, output_path)
            
    except Exception as e:
        warnings.warn(f"Error processing weights: {str(e)}")
        
    return weights

def main(model=None, save_path=None):
    """Process model or weights with kernel optimizations."""
    if model is None:
        return None
        
    try:
        # Apply kernel optimizations
        for name, param in model.named_parameters():
            if param.dim() >= 2:  # Only process matrices
                param.data = optimize_kernel_layout(param.data)
                
        # Convert to optimal precision
        precision = get_optimal_precision()
        model = setup_model_precision(model, precision, save_path)
        
    except Exception as e:
        warnings.warn(f"Error in model processing: {str(e)}")
        
    return model

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str)
    args = parser.parse_args()
    load_and_process_weights(args.input_path, args.output_path)
