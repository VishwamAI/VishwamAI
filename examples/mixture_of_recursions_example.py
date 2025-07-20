"""
Example usage of VishwamAI with Mixture of Recursions (MoR).

This example demonstrates how to configure and train a VishwamAI model
with Mixture of Recursions capabilities, based on the paper:
https://arxiv.org/abs/2507.10524
"""

import jax
import jax.numpy as jnp
from vishwamai.model import ModelConfig, VishwamAIModel, create_train_state
from vishwamai.training import TrainingConfig, CurriculumTrainer
from vishwamai.advanced_multimodal import MultimodalConfig, GemmaInspiredMultimodalTransformer


def create_mor_model_config():
    """Create a model configuration with Mixture of Recursions enabled."""
    
    config = ModelConfig(
        dim=2048,
        depth=24,
        heads=16,
        head_dim=128,
        vocab_size=50304,
        max_seq_len=4096,
        
        use_moe=True,
        expert_count=8,
        expert_capacity=4,
        
        use_recursion=True,
        max_recursion_depth=3,
        recursion_capacity=2,
        
        # Efficiency features
        use_flash_attention=True,
        use_grouped_query_attention=True,
        gqa_groups=8,
        use_rmsnorm=True,
        use_rotary_embeddings=True,
        
        # Hardware optimizations
        use_bfloat16=True,
        gradient_checkpointing=True,
        kernel_fusion=True
    )
    
    return config


def create_mor_training_config(model_config):
    """Create training configuration with recursion curriculum learning."""
    
    training_config = TrainingConfig(
        model_config=model_config,
        
        # Training hyperparameters
        learning_rate=1e-4,
        batch_size=32,
        total_steps=50000,
        warmup_steps=2000,
        
        use_curriculum=True,
        curriculum_stages=[
            {
                "name": "simple_recursion", 
                "steps": 10000, 
                "max_seq_len": 512,
                "max_recursion_depth": 1
            },
            {
                "name": "medium_recursion", 
                "steps": 20000, 
                "max_seq_len": 1024,
                "max_recursion_depth": 2
            },
            {
                "name": "complex_recursion", 
                "steps": 20000, 
                "max_seq_len": 2048,
                "max_recursion_depth": 3
            }
        ],
        
        gradient_clip_norm=1.0,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        
        # Mixed precision
        use_bfloat16=True,
        loss_scale=2.0**15,
        
        log_every=100,
        eval_every=1000,
        save_every=5000,
        checkpoint_dir="./checkpoints/mor_model"
    )
    
    return training_config


def create_multimodal_mor_config():
    """Create a multimodal configuration with Mixture of Recursions."""
    
    config = MultimodalConfig(
        vocab_size=262144,
        embed_dim=2560,
        num_heads=8,
        num_kv_heads=4,
        head_dim=256,
        
        vision_embed_dim=1024,
        image_size=800,
        patch_size=14,
        vision_layers=24,
        vision_heads=16,
        num_vision_tokens=256,
        
        cross_attn_layers=[4, 8, 12, 16, 20],
        fusion_type="adaptive_gate",
        
        use_recursion=True,
        max_recursion_depth=3,
        recursion_capacity=2,
        
        # Efficiency features
        use_sliding_window=True,
        sliding_window_size=1024,
        attn_logits_soft_cap=50.0,
        dropout=0.1,
        layer_norm_eps=1e-6
    )
    
    return config


def main():
    """Main example demonstrating MoR usage."""
    
    print("VishwamAI Mixture of Recursions Example")
    print("=" * 50)
    
    rng_key = jax.random.PRNGKey(42)
    
    print("Creating model configuration with Mixture of Recursions...")
    model_config = create_mor_model_config()
    
    # Create model
    print("Initializing VishwamAI model...")
    model = VishwamAIModel(model_config)
    
    # Create training state
    print("Creating training state...")
    train_state = create_train_state(model, model_config, rng_key)
    
    print("Setting up curriculum training with recursion progression...")
    training_config = create_mor_training_config(model_config)
    
    trainer = CurriculumTrainer(training_config)
    trainer.state = train_state
    
    print(f"Model parameters: {sum(x.size for x in jax.tree_leaves(train_state.params)):,}")
    print(f"Recursion enabled: {model_config.use_recursion}")
    print(f"Max recursion depth: {model_config.max_recursion_depth}")
    print(f"Recursion capacity: {model_config.recursion_capacity}")
    
    print("\nTesting forward pass...")
    dummy_input = jnp.ones((1, 512), dtype=jnp.int32)
    
    # Forward pass
    logits = model.apply(train_state.params, dummy_input, training=False)
    print(f"Output shape: {logits.shape}")
    print(f"Output dtype: {logits.dtype}")
    
    print("\nCreating multimodal model with MoR...")
    multimodal_config = create_multimodal_mor_config()
    multimodal_model = GemmaInspiredMultimodalTransformer(multimodal_config)
    
    multimodal_params = multimodal_model.init(
        rng_key, 
        jnp.ones((1, 512), dtype=jnp.int32),
        training=False
    )
    
    print(f"Multimodal model parameters: {sum(x.size for x in jax.tree_leaves(multimodal_params)):,}")
    
    print("\nMixture of Recursions setup complete!")
    print("Key features enabled:")
    print("- Dynamic recursion depth routing")
    print("- Token-level selective computation")
    print("- Curriculum learning with recursion progression")
    print("- Cross-modal attention with recursion support")
    print("- Memory-efficient recursive forward passes")


if __name__ == "__main__":
    main()
