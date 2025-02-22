"""
Advanced model capabilities demonstration for Vishwamai
"""
import torch
from vishwamai.model import (
    create_expert_model,
    create_parallel_model,
    ModelArgs,
    UnifiedConfig
)
from vishwamai.utils.t4_utils import (
    enable_t4_optimizations,
    get_device_capabilities,
    get_memory_stats
)

def demonstrate_moe_model():
    """Demonstrate Mixture of Experts model"""
    print("\n=== Mixture of Experts Model Demo ===")
    
    # Create expert model
    model = create_expert_model(
        num_experts=8,
        num_experts_per_token=2,
        expert_capacity=32,
        dtype="fp16",
        use_flash_attention=True
    )
    print(f"Created MoE model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test model with dummy input
    batch_size, seq_len = 4, 128
    inputs = {
        "input_ids": torch.randint(0, model.config.vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len)
    }
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        
    print("\nMoE Model Outputs:")
    print(f"Hidden states shape: {outputs['hidden_states'].shape}")
    if 'expert_outputs' in outputs:
        print("Expert usage statistics:")
        for expert_id, stats in outputs['expert_outputs']['expert_stats'].items():
            print(f"Expert {expert_id}: {stats['usage_count']} tokens, load balance: {stats['load_balance']:.3f}")

def demonstrate_parallel_model():
    """Demonstrate parallel model execution"""
    print("\n=== Parallel Model Demo ===")
    
    # Check if we have multiple GPUs
    if torch.cuda.device_count() < 2:
        print("Parallel demo requires multiple GPUs, skipping...")
        return
        
    # Create parallel model configuration
    config = ModelArgs(
        hidden_size=2048,
        intermediate_size=8192,
        num_attention_heads=32,
        num_hidden_layers=32,
        unified=UnifiedConfig(
            parallel=dict(
                tensor_parallel_size=torch.cuda.device_count(),
                sequence_parallel=True
            ),
            transformer=dict(
                use_flash_attention=True,
                use_parallel_attention=True
            )
        )
    )
    
    # Create and initialize model
    model = create_parallel_model(config=config)
    print(f"\nCreated parallel model across {torch.cuda.device_count()} GPUs")
    
    # Test parallel execution
    batch_size, seq_len = 16, 512
    inputs = {
        "input_ids": torch.randint(0, config.vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len)
    }
    
    # Measure memory before and after
    initial_mem = get_memory_stats()
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    final_mem = get_memory_stats()
    
    print("\nParallel Execution Results:")
    print(f"Output shape: {outputs['hidden_states'].shape}")
    print(f"Memory usage per GPU:")
    for gpu_id in range(torch.cuda.device_count()):
        mem_used = (final_mem['per_gpu'][gpu_id]['allocated'] - 
                   initial_mem['per_gpu'][gpu_id]['allocated']) / 1024**2
        print(f"GPU {gpu_id}: {mem_used:.1f}MB")

def demonstrate_advanced_features():
    """Demonstrate advanced model features"""
    print("\n=== Advanced Features Demo ===")
    
    # Create configuration with advanced features
    config = ModelArgs(
        hidden_size=1024,
        num_attention_heads=16,
        num_hidden_layers=24,
        unified=UnifiedConfig(
            transformer=dict(
                use_flash_attention=True,
                fused_qkv=True,
                fused_mlp=True,
                use_xformers=True,
                prenorm=True,
                use_rms_norm=True
            ),
            mlp=dict(
                gated_mlp=True,
                activation_fn="swiglu"
            )
        )
    )
    
    # Create model
    model = create_parallel_model(config=config)
    print("\nModel configuration:")
    print("- Flash Attention:", config.unified.transformer.use_flash_attention)
    print("- Fused operations:", config.unified.transformer.fused_qkv)
    print("- Gated MLP:", config.unified.mlp.gated_mlp)
    
    # Profile execution
    batch_size, seq_len = 8, 256
    inputs = {
        "input_ids": torch.randint(0, config.vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len)
    }
    
    # Warm up
    for _ in range(3):
        with torch.no_grad():
            model(**inputs)
            
    # Measure performance
    import time
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(10):
            outputs = model(**inputs)
            
    end_time = time.perf_counter()
    
    throughput = (10 * batch_size * seq_len) / (end_time - start_time)
    print(f"\nPerformance Results:")
    print(f"Throughput: {throughput:.0f} tokens/sec")
    print(f"Memory usage: {get_memory_stats()['allocated_gb']:.1f}GB")

def main():
    """Main demonstration function"""
    # Enable optimizations
    enable_t4_optimizations()
    
    # Print system info
    print("=== System Information ===")
    capabilities = get_device_capabilities()
    print(f"CUDA available: {capabilities['cuda_available']}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Flash Attention: {capabilities['flash_attention']}")
    print(f"BFloat16 support: {capabilities['bfloat16']}")
    print(f"Tensor Cores: {capabilities['tensor_cores']}")
    
    # Run demos
    try:
        demonstrate_moe_model()
    except Exception as e:
        print(f"MoE demo failed: {e}")
        
    try:
        demonstrate_parallel_model()
    except Exception as e:
        print(f"Parallel demo failed: {e}")
        
    try:
        demonstrate_advanced_features()
    except Exception as e:
        print(f"Advanced features demo failed: {e}")

if __name__ == "__main__":
    main()
