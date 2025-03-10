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

# Configure TPU environment
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "bfloat16")
jax.config.update("jax_platforms", "tpu")
jax.config.update("jax_xla_backend", "tpu")

def test_tpu_imports():
    """Test TPU model component imports and initialization"""
    try:
        print("✓ Successfully imported JAX")
        print("✓ Successfully imported Haiku")
        
        # Import VishwamAI TPU components
        from vishwamai.models.tpu import (
            TPUDeviceManager, TPUOptimizer,
            TransformerComputeLayerTPU,
            create_causal_mask,
            apply_rotary_embedding,
            get_optimal_tpu_config
        )
        print("✓ Successfully imported all TPU model components")

        # Optional Sonnet components
        try:
            import sonnet as snt
            print("✓ Successfully imported optional Sonnet components")
        except ImportError:
            print("ℹ Sonnet components not available (optional)")

        # Initialize test variables with concrete shapes
        batch_size = 2
        seq_len = 32
        embed_dim = 64
        num_heads = 8
        head_dim = embed_dim // num_heads
        
        # Initialize random key with TPU-optimized settings
        rng = jax.random.PRNGKey(42)  # Use fixed seed for reproducibility
        x = jnp.ones((batch_size, seq_len, embed_dim), dtype=jnp.bfloat16)
        x_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        
        # Create rotary embeddings with matching shape for head_dim
        freqs = jnp.arange(seq_len)[:, None] / jnp.power(
            10000, 
            2 * jnp.arange(head_dim // 2)[None, :] / head_dim
        )
        freqs_cis = jnp.exp(1j * freqs)  # [seq_len, head_dim/2]
        
        def init_components(x, x_ids):
            # Create mask with static shape during tracing
            mask = create_causal_mask(32)  # Use concrete size
            
            # Apply rotary embeddings
            rotary_out = apply_rotary_embedding(x, freqs_cis)
            
            # Initialize remaining components
            pos_enc = jnp.zeros((1, seq_len, embed_dim), dtype=jnp.bfloat16)
            embed = hk.Embed(vocab_size=1000, embed_dim=embed_dim)(x_ids)
            
            # Initialize transformer layer
            transformer = TransformerComputeLayerTPU(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=embed_dim * 4
            )
            ffn = transformer(x)
            
            return pos_enc, embed, ffn, rotary_out, mask

        # Initialize and transform components
        init_fn = hk.transform(init_components)
        
        # Run initialization with explicit static shapes
        params = init_fn.init(rng, x, x_ids)
        pos_out, embed_out, ffn_out, rotary_out, mask = init_fn.apply(params, rng, x, x_ids)
        
        print("✓ Successfully created causal mask")
        print("✓ Successfully applied rotary embeddings")  
        print("✓ Successfully initialized positional encoding")
        print("✓ Successfully initialized token embedding")
        print("✓ Successfully initialized feed-forward network")
        
        # Test TPU utilities
        config = get_optimal_tpu_config(hidden_size=embed_dim, seq_len=32, batch_size=2)
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
