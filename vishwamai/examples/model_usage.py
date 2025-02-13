import torch
from pathlib import Path
from vishwamai.model_utils import load_model, print_model_size

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model with optimized configuration
    config_path = Path(__file__).parent.parent / "configs" / "config_optimized.json"
    print("\nLoading model from config:", config_path)
    
    # Prepare configuration overrides
    overrides = {
        'cache_augmentation.max_length': 2048,
        'neural_memory.memory_size': 768,
        'tree_of_thoughts.max_depth': 4
    }
    
    # Load model with overrides
    model = load_model(
        config_path=config_path,
        device=device,
        **overrides
    )
    
    print("\nModel loaded successfully!")
    print_model_size(model)
    
    # Example input
    batch_size = 1
    seq_length = 128
    input_ids = torch.randint(
        0,
        model.args.vocab_size,
        (batch_size, seq_length),
        device=device
    )
    
    print(f"\nProcessing input sequence of length {seq_length}")
    
    # Generate output
    with torch.inference_mode():
        output = model(input_ids)
    
    print("\nOutput shape:", output.shape)
    print("Output dtype:", output.dtype)
    
    # Example of memory reset
    print("\nResetting memory states...")
    for layer in model.layers:
        if hasattr(layer, 'memory'):
            layer.memory.reset_memory()
        if hasattr(layer, 'cache'):
            layer.cache.reset_cache()
    
    print("\nDone!")

if __name__ == "__main__":
    main()
