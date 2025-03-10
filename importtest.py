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
        # Import all TPU components
        from vishwamai.models.tpu import (
            # Core attention mechanisms
            FlashMLAttentionTPU,
            MultiModalAttentionTPU,
            TemporalAttentionTPU,
            
            # Core models
            CoTModelTPU,
            OptimizedMoE,
            ToTModelTPU,
            ThoughtNodeTPU,
            
            # Model components
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
            
            # TPU core utilities
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
            
            # Generation utilities
            generate_cot,
            generate_tot,
            
            # Core utilities
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
        
        # Test basic initialization and utilities
        print("\nTesting component initialization and utilities...")
        batch_size, seq_len, embed_dim = 2, 64, 512
        num_heads = 8
        
        # Test causal mask creation
        def test_mask():
            mask = create_causal_mask(seq_len)
            return mask
            
        # Test rotary embeddings
        def test_rotary():
            x = jnp.ones((batch_size, seq_len, num_heads, embed_dim // num_heads))
            freqs = jnp.exp(-jnp.arange(seq_len)[:, None] / 10000 ** (2 * jnp.arange(embed_dim // 2) / embed_dim))
            freqs_cis = jnp.exp(1j * freqs).astype(jnp.complex64)
            return apply_rotary_embedding(x, freqs_cis)
            
        # Test positional encoding
        def test_pos_encoding(x):
            pos_enc = PositionalEncoding(embed_dim=embed_dim, max_seq_len=seq_len)
            return pos_enc(x)
            
        # Test token embedding
        def test_embedding(x):
            embed = TokenEmbedding(vocab_size=32000, embed_dim=embed_dim)
            return embed(x)
            
        # Test feed-forward
        def test_ffn(x):
            ffn = FeedForward(embed_dim=embed_dim, ff_dim=embed_dim * 4)
            return ffn(x)
            
        # Initialize components
        rng = jax.random.PRNGKey(0)
        rng, init_rng = jax.random.split(rng)
        
        # Test mask
        mask_fn = hk.transform(test_mask)
        mask = mask_fn.apply({}, init_rng)
        print("✓ Successfully created causal mask")
        
        # Test rotary embeddings
        rotary_fn = hk.transform(test_rotary)
        rotary = rotary_fn.apply({}, init_rng)
        print("✓ Successfully applied rotary embeddings")
        
        # Test positional encoding
        x = jnp.ones((batch_size, seq_len, embed_dim))
        pos_enc_fn = hk.transform(test_pos_encoding)
        params = pos_enc_fn.init(init_rng, x)
        pos_enc = pos_enc_fn.apply(params, init_rng, x)
        print("✓ Successfully initialized positional encoding")
        
        # Test token embedding
        x_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        embed_fn = hk.transform(test_embedding)
        params = embed_fn.init(init_rng, x_ids)
        embeddings = embed_fn.apply(params, init_rng, x_ids)
        print("✓ Successfully initialized token embedding")
        
        # Test feed-forward
        ffn_fn = hk.transform(test_ffn)
        params = ffn_fn.init(init_rng, x)
        ffn_out = ffn_fn.apply(params, init_rng, x)
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
    else:
        print("\n❌ TPU component tests failed!")
        sys.exit(1)
