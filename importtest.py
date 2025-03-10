"""
TPU-optimized model components test
"""
import os
import sys
import warnings
import jax
import jax.numpy as jnp
import haiku as hk
import math

def test_tpu_imports():
    print("Testing TPU model component imports...")
    
    try:
        from vishwamai.models.tpu import (
            # Core attention mechanisms
            FlashMLAttentionTPU,
            MultiModalAttentionTPU,
            TemporalAttentionTPU,
            
            # Core models and components
            CoTModelTPU,
            OptimizedMoE,
            ToTModelTPU,
            ThoughtNodeTPU,
            TransformerComputeLayerTPU,
            TransformerMemoryLayerTPU,
            HybridThoughtAwareAttentionTPU,
            PositionalEncoding,
            TokenEmbedding,
            FeedForward,
            
            # Expert components
            ExpertModule,
            ExpertRouter,
            ExpertGating,
            
            # Core utilities
            TPUDeviceManager,
            TPUOptimizer,
            TPUDataParallel,
            TPUProfiler,
            TPUModelUtils,
            apply_rotary_embedding,
            create_causal_mask,
            
            # Kernel layers
            TPUGEMMLinear,
            TPUGroupedGEMMLinear,
            TPULayerNorm,
            DeepGEMMLinear,
            DeepGEMMLayerNorm,
            DeepGEMMGroupedLinear,
            gelu_kernel,
            
            # Utilities
            generate_cot,
            generate_tot,
            compute_load_balancing_loss,
            get_optimal_tpu_config,
            benchmark_matmul,
            compute_numerical_error
        )
        print("✓ Successfully imported all TPU model components")

        # Try importing optional Sonnet components
        try:
            from vishwamai.models.tpu import SonnetFlashAttentionTPU
            print("✓ Successfully imported optional Sonnet components")
        except ImportError:
            warnings.warn("Sonnet components not available. This is optional and won't affect core functionality.")
        
        print("\nTesting component initialization and utilities...")
        batch_size, seq_len, embed_dim = 2, 64, 512
        num_heads = 8
        
        # Initialize test data
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((batch_size, seq_len, embed_dim))
        x_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        x_rot = jnp.ones((batch_size, seq_len, num_heads, embed_dim // num_heads))
        
        def init_components(x, x_ids):
            # Initialize core components
            pos_enc = PositionalEncoding(embed_dim=embed_dim, max_seq_len=seq_len)
            embed = TokenEmbedding(vocab_size=32000, embed_dim=embed_dim)
            ffn = FeedForward(embed_dim=embed_dim, ff_dim=embed_dim * 4)
            
            # Run forward passes
            mask = create_causal_mask(seq_len)
            freqs = jnp.exp(-jnp.arange(seq_len)[:, None] / 10000 ** (2 * jnp.arange(embed_dim // 2) / embed_dim))
            freqs_cis = jnp.exp(1j * freqs).astype(jnp.complex64)
            
            pos_out = pos_enc(x)
            embed_out = embed(x_ids)
            ffn_out = ffn(x)
            rotary_out = apply_rotary_embedding(x_rot, freqs_cis)
            
            return pos_out, embed_out, ffn_out, rotary_out, mask
        
        # Initialize and transform components
        init_fn = hk.transform(init_components)
        
        # Run initialization
        params = init_fn.init(rng, x, x_ids)
        pos_out, embed_out, ffn_out, rotary_out, mask = init_fn.apply(params, rng, x, x_ids)
        
        print("✓ Successfully created causal mask")
        print("✓ Successfully applied rotary embeddings")
        print("✓ Successfully initialized positional encoding")
        print("✓ Successfully initialized token embedding")
        print("✓ Successfully initialized feed-forward network")
        
        # Test TPU utilities
        config = get_optimal_tpu_config(hidden_size=embed_dim, seq_len=seq_len, batch_size=batch_size)
        print("✓ Successfully generated TPU configuration")
        
        # Test hardware detection if available
        try:
            capabilities = TPUDeviceManager.get_hardware_capabilities()
            print(f"✓ Detected hardware: {capabilities['device_type']}")
        except Exception as e:
            print("ℹ Hardware detection skipped in this environment")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("\nPlease ensure you have installed the required packages:")
        print("pip install jax jaxlib dm-haiku optax")
        return False
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        print("\nError details:", str(e))
        return False

if __name__ == "__main__":
    print("Python version:", sys.version)
    print("JAX version:", jax.__version__)
    print("Haiku version:", hk.__version__)
    print("\nStarting TPU component tests...")
    
    success = test_tpu_imports()
    if success:
        print("\n✅ All TPU component tests passed!")
        print("\nComponent shapes:")
        print("- Causal mask:", "(seq_len, seq_len)")
        print("- Positional encoding:", "(batch_size, seq_len, embed_dim)")
        print("- Token embedding:", "(batch_size, seq_len, embed_dim)")
        print("- Feed-forward:", "(batch_size, seq_len, embed_dim)")
        print("- Rotary embedding:", "(batch_size, seq_len, num_heads, head_dim)")
    else:
        print("\n❌ TPU component tests failed!")
        sys.exit(1)
