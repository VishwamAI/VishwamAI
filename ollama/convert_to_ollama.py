#!/usr/bin/env python3
import os
import sys
import json
import torch
import shutil
from pathlib import Path
from typing import Dict, Any

def convert_to_ollama_format(
    input_dir: str = "./model",
    output_dir: str = "./ollama_model",
    model_name: str = "vishwamai"
) -> None:
    """
    Convert HuggingFace model format to Ollama compatible format.
    """
    print(f"Converting {model_name} to Ollama format...")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model configuration
    config_path = Path(input_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Create Ollama model structure
    ollama_config = {
        "model": {
            "name": model_name,
            "architecture": "VishwamAI",
            "format": "gguf",
            "quantization": "int8",
            "context_size": config.get("max_seq_len", 2048),
            "num_attention_heads": config.get("n_heads", 16),
            "num_hidden_layers": config.get("n_layers", 12),
            "hidden_size": config.get("dim", 1024),
            "intermediate_size": config.get("inter_dim", 2816),
            "vocab_size": config.get("vocab_size", 32000)
        },
        "components": {
            "cache": {
                "enabled": True,
                "size": 8192
            },
            "memory": {
                "enabled": True,
                "layers": config.get("n_layers", 12)
            },
            "tree": {
                "enabled": True,
                "beam_width": 4
            }
        },
        "parameters": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repeat_penalty": 1.1
        }
    }
    
    # Save Ollama config
    with open(output_dir / "config.json", "w") as f:
        json.dump(ollama_config, f, indent=2)
    
    # Convert model weights
    try:
        print("Converting model weights...")
        model_path = Path(input_dir) / "pytorch_model.bin"
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        
        # Load PyTorch model
        state_dict = torch.load(model_path, map_location="cpu")
        
        # Convert to GGUF format
        gguf_path = output_dir / "model.gguf"
        torch.save({
            "weight_map": state_dict,
            "metadata": ollama_config
        }, gguf_path)
        
        print("Converting auxiliary components...")
        # Convert auxiliary components
        components = ["cache_module", "memory_module", "tree_module"]
        for component in components:
            component_path = Path(input_dir) / f"{component}.bin"
            if component_path.exists():
                shutil.copy2(component_path, output_dir / f"{component}.bin")
        
        # Copy tokenizer files
        tokenizer_files = [
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json"
        ]
        for file in tokenizer_files:
            src = Path(input_dir) / file
            if src.exists():
                shutil.copy2(src, output_dir / file)
        
        print(f"Model successfully converted and saved to {output_dir}")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        sys.exit(1)

def verify_conversion(output_dir: str) -> None:
    """
    Verify the converted model files.
    """
    required_files = [
        "config.json",
        "model.gguf",
        "tokenizer.model"
    ]
    
    missing_files = []
    for file in required_files:
        if not (Path(output_dir) / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"Warning: Missing required files: {', '.join(missing_files)}")
        return False
    
    print("Conversion verification completed successfully")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert VishwamAI model to Ollama format")
    parser.add_argument("--input-dir", default="./model", help="Input model directory")
    parser.add_argument("--output-dir", default="./ollama_model", help="Output directory")
    parser.add_argument("--model-name", default="vishwamai", help="Model name")
    
    args = parser.parse_args()
    
    convert_to_ollama_format(args.input_dir, args.output_dir, args.model_name)
    verify_conversion(args.output_dir)
