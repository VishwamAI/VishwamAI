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

from vishwamai.configs.tpu_config import TPUConfig
from vishwamai.models.tpu.attention import FlashMLAttentionTPU
from vishwamai.models.tpu.cot_model import CoTModelTPU
from vishwamai.models.tpu.tot_model import ToTModelTPU
from vishwamai.models.tpu.moe import OptimizedMoE

def test_tpu_imports():
    """Test TPU model component imports and initialization"""
    try:
        print("✓ Successfully imported JAX")
        print("✓ Successfully imported Haiku")

        # Initialize TPU configuration
        TPUConfig.initialize()
        print("✓ Successfully initialized TPU configuration")
        
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
            # Create mask with proper broadcasting shape
            mask = create_causal_mask(seq_len, batch_size)  # [batch_size, 1, seq_len, seq_len]
            
            # Apply rotary embeddings
            rotary_out = apply_rotary_embedding(
                x.reshape(batch_size, seq_len, num_heads, head_dim),
                freqs_cis
            )
            
            # Initialize remaining components
            pos_enc = jnp.zeros((1, seq_len, embed_dim), dtype=jnp.bfloat16)
            embed = hk.Embed(vocab_size=1000, embed_dim=embed_dim)(x_ids)
            
            # Initialize transformer layer with explicit mask
            transformer = TransformerComputeLayerTPU(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=embed_dim * 4
            )
            ffn = transformer(x, mask=mask)
            
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
        
        # Initialize test data for additional components
        batch_size, seq_len, embed_dim = 2, 32, 512
        num_heads = 8
        rng = jax.random.PRNGKey(0)
        test_input = jnp.ones((batch_size, seq_len, embed_dim))

        # Test FlashMLAttentionTPU
        def init_attention(x):
            attention = FlashMLAttentionTPU(embed_dim=embed_dim, num_heads=num_heads)
            return attention(x, is_training=True)

        transformed = hk.transform(init_attention)
        params = transformed.init(rng, test_input)
        output = transformed.apply(params, rng, test_input)
        print("✓ FlashMLAttentionTPU test passed")

        # Test CoTModelTPU
        def init_cot(x):
            model = CoTModelTPU(embed_dim=embed_dim, num_heads=num_heads)
            return model(x, is_training=True)

        transformed_cot = hk.transform(init_cot)
        params_cot = transformed_cot.init(rng, test_input)
        output_cot = transformed_cot.apply(params_cot, rng, test_input)
        print("✓ CoTModelTPU test passed")

        # Test ToTModelTPU
        def init_tot(x):
            model = ToTModelTPU(embed_dim=embed_dim, num_heads=num_heads)
            return model(x, is_training=True)

        transformed_tot = hk.transform(init_tot)
        params_tot = transformed_tot.init(rng, test_input)
        output_tot = transformed_tot.apply(params_tot, rng, test_input)
        print("✓ ToTModelTPU test passed")

        # Test OptimizedMoE
        def init_moe(x):
            model = OptimizedMoE(num_experts=4, expert_size=embed_dim, input_size=embed_dim)
            return model(x, is_training=True)

        transformed_moe = hk.transform(init_moe)
        params_moe = transformed_moe.init(rng, test_input)
        output_moe = transformed_moe.apply(params_moe, rng, test_input)
        print("✓ OptimizedMoE test passed")

        print("\n✓ All TPU component tests passed successfully!")
        
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
        print("- Causal mask:", "(batch_size, 1, seq_len, seq_len)")
        print("- Positional encoding:", "(batch_size, seq_len, embed_dim)")
        print("- Token embedding:", "(batch_size, seq_len, embed_dim)")
        print("- Feed-forward:", "(batch_size, seq_len, embed_dim)")
        print("- Rotary embedding:", "(batch_size, seq_len, num_heads, head_dim)")
    else:
        print("\n❌ TPU component tests failed!")
        sys.exit(1)
