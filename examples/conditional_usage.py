"""Example usage of conditional computation features in VishwamAI."""

import jax
import jax.numpy as jnp
from vishwamai import VishwamAI
from vishwamai.layers import (
    DynamicChannelGating,
    ConditionalInfoGainNode,
    CIGTLayer,
    RLBasedConditionalLayer
)

def main():
    # Create model with conditional computation
    conditional_config = {
        'use_cign': True,
        'use_cigt': True,
        'use_rl': True,
        'gating_dim': 256,
        'num_splits': 2,
        'num_paths': 4,
        'num_experts': 8
    }
    
    model = VishwamAI(
        vocab_size=32000,
        hidden_size=4096,
        num_layers=32,
        conditional_config=conditional_config
    )
    
    # Example input
    batch_size = 4
    seq_len = 128
    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    
    # Initialize model
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, input_ids, training=True)
    
    # Forward pass with conditional computation
    outputs = model.apply(
        params,
        input_ids,
        training=True,
        rngs={'dropout': rng}
    )
    
    # Extract and analyze conditional computation metrics
    print("\nConditional Computation Analysis:")
    print("----------------------------------")
    
    if outputs['gates'] is not None:
        gate_stats = {
            'mean_activation': float(jnp.mean(outputs['gates'])),
            'sparsity': float(jnp.mean(outputs['gates'] < 0.1))
        }
        print("\nDynamic Gating Statistics:")
        print(f"Mean activation: {gate_stats['mean_activation']:.3f}")
        print(f"Sparsity: {gate_stats['sparsity']:.3f}")
    
    if outputs['split_probs'] is not None:
        print("\nCIGN Routing Probabilities:")
        split_stats = {
            'entropy': float(-jnp.sum(
                outputs['split_probs'] * jnp.log(outputs['split_probs'] + 1e-10),
                axis=-1
            ).mean())
        }
        print(f"Route entropy: {split_stats['entropy']:.3f}")
    
    if outputs['info_state'] is not None:
        print("\nCIGT Information Flow:")
        info_stats = {
            'mean_info': float(jnp.mean(outputs['info_state'])),
            'active_paths': float(jnp.mean(outputs['info_state'] > 0.1))
        }
        print(f"Mean information: {info_stats['mean_info']:.3f}")
        print(f"Active paths: {info_stats['active_paths']:.3f}")
    
    if outputs['rl_metrics'] is not None:
        print("\nRL-based Layer Metrics:")
        policy_loss = float(outputs['rl_metrics'].get('policy_loss', 0.0))
        action_entropy = float(-jnp.sum(
            outputs['rl_metrics']['actions'] * 
            jnp.log(outputs['rl_metrics']['actions'] + 1e-10),
            axis=-1
        ).mean())
        print(f"Policy loss: {policy_loss:.3f}")
        print(f"Action entropy: {action_entropy:.3f}")
    
    print("\nModel Performance:")
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Number of layer outputs: {len(outputs['layer_outputs'])}")

if __name__ == "__main__":
    main()