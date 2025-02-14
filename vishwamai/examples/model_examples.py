#!/usr/bin/env python3
"""
Examples of using VishwamAI model in different environments.
"""

import torch
from vishwamai.model_utils import load_model

def basic_usage():
    """Basic model usage example."""
    # Load model
    model = load_model(
        config_path="configs/config_optimized.json",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Create input
    batch_size = 1
    seq_length = 128
    tokens = torch.randint(0, model.args.vocab_size, (batch_size, seq_length))
    tokens = tokens.to(model.device)
    
    # Generate output
    with torch.inference_mode():
        output = model(tokens)
        
    print(f"Input shape: {tokens.shape}")
    print(f"Output shape: {output.shape}")
    return output

def memory_efficient_usage():
    """Memory-efficient usage example."""
    model = load_model(
        config_path="configs/config_optimized.json",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_cache=False  # Disable caching for memory efficiency
    )
    
    # Enable gradient checkpointing if needed
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    # Process longer sequences in chunks
    chunk_size = 512
    full_seq_length = 2048
    batch_size = 1
    
    tokens = torch.randint(0, model.args.vocab_size, (batch_size, full_seq_length))
    tokens = tokens.to(model.device)
    
    # Process in chunks
    outputs = []
    for i in range(0, full_seq_length, chunk_size):
        chunk = tokens[:, i:i+chunk_size]
        with torch.inference_mode():
            output = model(chunk)
        outputs.append(output)
        
    print(f"Processed {full_seq_length} tokens in chunks of {chunk_size}")
    return outputs

def cpu_fallback_usage():
    """Example with CPU fallback."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(
            config_path="configs/config_optimized.json",
            device=device
        )
        
        # Small input for testing
        tokens = torch.randint(0, model.args.vocab_size, (1, 64))
        tokens = tokens.to(device)
        
        with torch.inference_mode():
            output = model(tokens)
            
        print(f"Successfully ran on {device}")
        return output
        
    except RuntimeError as e:
        print(f"GPU error: {e}")
        print("Falling back to CPU...")
        
        # Try again on CPU
        model = load_model(
            config_path="configs/config_optimized.json",
            device="cpu"
        )
        
        tokens = torch.randint(0, model.args.vocab_size, (1, 64))
        with torch.inference_mode():
            output = model(tokens)
            
        print("Successfully ran on CPU")
        return output

def main():
    """Run examples."""
    print("\n1. Basic Usage Example:")
    basic_usage()
    
    print("\n2. Memory Efficient Usage Example:")
    memory_efficient_usage()
    
    print("\n3. CPU Fallback Example:")
    cpu_fallback_usage()

if __name__ == "__main__":
    main()
