"""
Example usage of VishwamAI's TPU-optimized transformer
"""

import jax
import jax.numpy as jnp
import json
from functools import partial
from vishwamai.transformer import (
    EnhancedTransformerModel,
    DTYPE_CONFIG,
    TPUGEMMLinear,
    TPULayerNorm
)
from vishwamai.device_mesh import TPUMeshContext
from vishwamai.kernels.tpu.tpu_custom_call import optimize_tpu_layout, pad_to_tpu_multiple

def load_config(config_path):
    """Load transformer configuration"""
    with open(config_path) as f:
        return json.load(f)

def create_transformer_model(config):
    """Create a transformer model instance with TPU optimizations"""
    arch_config = config['architecture']
    
    # Ensure dimensions are optimal for TPU (multiples of 128)
    hidden_dim = ((arch_config['hidden_size'] + 127) // 128) * 128
    mlp_dim = ((arch_config['intermediate_size'] + 127) // 128) * 128
    
    return EnhancedTransformerModel(
        vocab_size=arch_config['vocab_size'],
        num_layers=arch_config['num_hidden_layers'],
        num_heads=arch_config['num_attention_heads'],
        head_dim=arch_config['head_dim'],
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        max_seq_len=arch_config['max_position_embeddings'],
        dropout_rate=arch_config['dropout_rate'],
        use_flash_attn=True,  # Enable TPU-optimized flash attention
        use_rms_norm=True,    # Use more efficient RMSNorm
        dtype=jnp.bfloat16    # Use TPU native bfloat16
    )

def setup_tpu_mesh():
    """Setup TPU device mesh for efficient parallel processing"""
    devices = jax.devices()
    config = {
        "tpu": {
            "tpu_cores": len(devices),
            "device_strategy": "data_parallel"
        }
    }
    return TPUMeshContext(config, data_parallel=True)

def generate_text(model, params, input_ids, max_length=100, chunk_size=32):
    """Generate text using the transformer model with TPU optimizations"""
    # Ensure input has correct shape and optimal layout
    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]
    
    # Pad input to TPU-friendly size
    padded_input = pad_to_tpu_multiple(input_ids, multiple=128)
    batch_size = padded_input.shape[0]
    
    # Process in chunks for memory efficiency
    generated = padded_input
    for _ in range(max_length - padded_input.shape[1]):
        # Process sequence in chunks
        logits_chunks = []
        for i in range(0, generated.shape[1], chunk_size):
            chunk = jax.lax.dynamic_slice(
                generated,
                (0, i),
                (batch_size, min(chunk_size, generated.shape[1] - i))
            )
            # Optimize chunk layout for TPU
            chunk = optimize_tpu_layout(chunk)
            
            # Get logits for chunk
            chunk_logits = model.apply(
                {'params': params},
                chunk,
                deterministic=True
            )
            logits_chunks.append(chunk_logits)
        
        # Combine chunks and get next token
        logits = jnp.concatenate(logits_chunks, axis=1)
        next_token = jnp.argmax(logits[:, -1, :], axis=-1)
        generated = jnp.concatenate([generated, next_token[:, None]], axis=1)
    
    # Remove padding
    original_batch_size = input_ids.shape[0]
    return generated[:original_batch_size]

def main():
    # Load configuration
    config = load_config('vishwamai/configs/transformer_config.json')
    
    # Setup TPU mesh
    mesh_context = setup_tpu_mesh()
    
    # Initialize model within TPU mesh context
    with mesh_context.mesh_context():
        model = create_transformer_model(config)
        
        # Initialize with optimal TPU layout
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, 128), dtype=jnp.int32)  # Pad to 128 for TPU efficiency
        variables = model.init(rng, dummy_input)
        
        # Example input sequence
        input_sequence = jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.int32)
        
        # Generate text with TPU optimizations
        output_sequence = generate_text(
            model,
            variables['params'],
            input_sequence,
            max_length=50,
            chunk_size=32  # Process in 32-token chunks
        )
        
        print(f"Input shape: {input_sequence.shape}")
        print(f"Output shape: {output_sequence.shape}")

if __name__ == "__main__":
    main()