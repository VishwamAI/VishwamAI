#!/usr/bin/env python3
"""
Example script showing how to use the VishwamAI model.
"""

import torch
from vishwamai.model_utils import load_model, get_gpu_memory

def main():
    # Get GPU information
    gpu_available = torch.cuda.is_available()
    device = "cuda" if gpu_available else "cpu"
    
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = get_gpu_memory()
        print(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("Using CPU")

    # Load model with optimized settings
    model = load_model(
        config_path="configs/config_optimized.json",
        device=device
    )
    print(f"Model loaded with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

    # Example input
    input_text = "What is 2+2?"
    tokens = torch.randint(0, model.args.vocab_size, (1, 128)).to(device)  # Placeholder for actual tokenization
    
    # Generate output
    with torch.inference_mode():
        output = model(tokens)
    print(f"Output shape: {output.shape}")

    # Example of saving model
    torch.save(model.state_dict(), "model_checkpoint.pt")
    print("Model saved to model_checkpoint.pt")

    # Load saved model
    new_model = load_model(
        config_path="configs/config_optimized.json",
        device=device,
        pretrained_path="model_checkpoint.pt"
    )
    print("Model loaded from checkpoint")

if __name__ == "__main__":
    main()
