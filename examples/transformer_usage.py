"""
Example usage of VishwamAI's TPU-optimized transformer
"""

import jax
import jax.numpy as jnp
import json
from vishwamai.transformer import (
    TransformerModel,
    DTYPE_CONFIG,
    TokenEmbedding
)

def load_config(config_path):
    """Load transformer configuration"""
    with open(config_path) as f:
        return json.load(f)

def create_transformer_model(config):
    """Create a transformer model instance from config"""
    arch_config = config['architecture']
    
    return TransformerModel(
        vocab_size=arch_config['vocab_size'],
        num_layers=arch_config['num_hidden_layers'],
        num_heads=arch_config['num_attention_heads'],
        head_dim=arch_config['head_dim'],
        hidden_dim=arch_config['hidden_size'],
        mlp_dim=arch_config['intermediate_size'],
        max_seq_len=arch_config['max_position_embeddings'],
        dropout_rate=arch_config['dropout_rate'],
        dtype=DTYPE_CONFIG['compute_dtype']
    )

def generate_text(model, params, input_ids, max_length=100):
    """Generate text using the transformer model"""
    # Ensure input has correct shape
    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]
        
    batch_size = input_ids.shape[0]
    generated = input_ids
    
    # Generate tokens auto-regressively
    for _ in range(max_length - input_ids.shape[1]):
        # Get logits for next token
        logits = model.apply(
            {'params': params},
            generated,
            deterministic=True
        )
        next_token = jnp.argmax(logits[:, -1, :], axis=-1)
        generated = jnp.concatenate([generated, next_token[:, None]], axis=1)
    
    return generated

def main():
    # Load configuration
    config = load_config('vishwamai/configs/transformer_config.json')
    
    # Initialize model
    model = create_transformer_model(config)
    
    # Initialize parameters with dummy input
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 16), dtype=jnp.int32)
    variables = model.init(rng, dummy_input)
    
    # Example input sequence (replace with actual tokenized input)
    input_sequence = jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.int32)
    
    # Generate text
    output_sequence = generate_text(
        model,
        variables['params'],
        input_sequence,
        max_length=50
    )
    
    print(f"Input shape: {input_sequence.shape}")
    print(f"Output shape: {output_sequence.shape}")
    
    # In practice, you would decode the tokens back to text here
    # using your tokenizer

if __name__ == "__main__":
    main()