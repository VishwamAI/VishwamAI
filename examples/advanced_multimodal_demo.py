"""
Advanced Multimodal VishwamAI - Complete Example

This script demonstrates the complete new formulation and techniques
for building multimodal AI systems inspired by Google DeepMind's Gemma
architecture with advanced Q, K, V attention mechanisms.

Features Demonstrated:
- Gemma-inspired multimodal transformer
- Grouped Query Attention (GQA)
- SigLIP-style vision encoder
- Cross-modal attention and fusion
- Advanced training pipeline with curriculum learning
- Memory-efficient Flash Attention
- Progressive multimodal training
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path

# Import our new multimodal components
from vishwamai.advanced_multimodal import (
    GemmaInspiredMultimodalTransformer,
    MultimodalConfig,
    GEMMA_4B_MULTIMODAL_CONFIG,
    GEMMA_12B_MULTIMODAL_CONFIG,
    create_multimodal_model
)

from vishwamai.gemma_attention import (
    FlashAttention2,
    CrossModalAttention,
    GroupedQueryAttention,
    AttentionType,
    QueryPreAttentionNorm,
    create_attention_layer,
    GEMMA_ATTENTION_CONFIG
)

from vishwamai.multimodal_training import (
    MultimodalTrainer,
    TrainingConfig,
    CurriculumStage,
    SMALL_MODEL_TRAINING_CONFIG,
    MEDIUM_MODEL_TRAINING_CONFIG,
    LARGE_MODEL_TRAINING_CONFIG,
    main_training_pipeline
)


def setup_logging():
    """Setup logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def create_dummy_multimodal_data(
    batch_size: int = 8,
    seq_length: int = 512,
    vocab_size: int = 32000,
    image_size: int = 224,
    num_batches: int = 100
):
    """Create dummy multimodal data for demonstration."""
    
    def data_generator():
        """Generator for dummy multimodal batches."""
        rng = jax.random.PRNGKey(42)
        
        for i in range(num_batches):
            rng, batch_rng = jax.random.split(rng)
            
            # Generate text data
            input_ids = jax.random.randint(
                batch_rng,
                (batch_size, seq_length),
                minval=0,
                maxval=vocab_size
            )
            
            attention_mask = jnp.ones((batch_size, seq_length), dtype=jnp.bool_)
            
            # Generate image data
            images = jax.random.normal(
                batch_rng,
                (batch_size, image_size, image_size, 3)
            )
            
            # Normalize images to [0, 1] range
            images = (images - jnp.min(images)) / (jnp.max(images) - jnp.min(images))
            images = (images * 255).astype(jnp.uint8)
            
            yield {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': input_ids,  # For language modeling
                'images': images,
                'loss_mask': attention_mask
            }
    
    return data_generator()


def demonstrate_attention_mechanisms():
    """Demonstrate different attention mechanisms."""
    
    logger = setup_logging()
    logger.info("🔍 Demonstrating Advanced Attention Mechanisms")
    
    # Configuration
    embed_dim = 1024
    num_heads = 16
    num_kv_heads = 8
    seq_length = 512
    batch_size = 4
    
    # Create dummy input
    rng = jax.random.PRNGKey(0)
    hidden_states = jax.random.normal(rng, (batch_size, seq_length, embed_dim))
    
    logger.info(f"Input shape: {hidden_states.shape}")
    
    # 1. Flash Attention with Grouped Query Attention
    logger.info("\n1. Flash Attention 2 with Grouped Query Attention")
    
    flash_attention = FlashAttention2(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=embed_dim // num_heads,
        attn_type=AttentionType.GLOBAL,
        use_qk_norm=True,
        attn_logits_soft_cap=50.0
    )
    
    variables = flash_attention.init(rng, hidden_states=hidden_states)
    flash_output = flash_attention.apply(variables, hidden_states=hidden_states)
    
    logger.info(f"Flash Attention output shape: {flash_output.shape}")
    logger.info(f"Memory reduction with GQA: {num_heads // num_kv_heads}x")
    
    # 2. Sliding Window Attention
    logger.info("\n2. Sliding Window Attention")
    
    sliding_attention = FlashAttention2(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=embed_dim // num_heads,
        attn_type=AttentionType.LOCAL_SLIDING,
        window_size=128,
        use_qk_norm=True
    )
    
    variables = sliding_attention.init(rng, hidden_states=hidden_states)
    sliding_output = sliding_attention.apply(variables, hidden_states=hidden_states)
    
    logger.info(f"Sliding Window Attention output shape: {sliding_output.shape}")
    logger.info("Window size: 128 tokens (vs full 512 for global)")
    
    # 3. Cross-Modal Attention
    logger.info("\n3. Cross-Modal Attention")
    
    vision_features = jax.random.normal(rng, (batch_size, 256, 768))  # Vision tokens
    
    cross_attention = CrossModalAttention(
        text_dim=embed_dim,
        vision_dim=768,
        num_heads=12
    )
    
    variables = cross_attention.init(
        rng, 
        text_features=hidden_states, 
        vision_features=vision_features
    )
    
    text_attended, vision_attended = cross_attention.apply(
        variables,
        text_features=hidden_states,
        vision_features=vision_features
    )
    
    logger.info(f"Cross-modal text output: {text_attended.shape}")
    logger.info(f"Cross-modal vision output: {vision_attended.shape}")
    
    # 4. Multi-Scale Attention
    logger.info("\n4. Multi-Scale Attention")
    
    multi_scale_attention = create_attention_layer(
        attention_type="multi_scale",
        embed_dim=embed_dim,
        num_heads=num_heads,
        window_sizes=[64, 256, 1024, -1]  # Different scales
    )
    
    variables = multi_scale_attention.init(rng, hidden_states=hidden_states)
    multi_scale_output = multi_scale_attention.apply(
        variables, 
        hidden_states=hidden_states
    )
    
    logger.info(f"Multi-scale attention output: {multi_scale_output.shape}")
    logger.info("Captures both local and global dependencies simultaneously")


def demonstrate_multimodal_model():
    """Demonstrate the complete multimodal model."""
    
    logger = setup_logging()
    logger.info("🤖 Demonstrating Gemma-Inspired Multimodal Model")
    
    # Create model with Gemma 4B configuration
    config = GEMMA_4B_MULTIMODAL_CONFIG
    model = GemmaInspiredMultimodalTransformer(config=config)
    
    logger.info(f"Model Configuration:")
    logger.info(f"  - Embedding dimension: {config.embed_dim}")
    logger.info(f"  - Attention heads: {config.num_heads}")
    logger.info(f"  - KV heads (GQA): {config.num_kv_heads}")
    logger.info(f"  - Vision embedding dim: {config.vision_embed_dim}")
    logger.info(f"  - Vision tokens: {config.num_vision_tokens}")
    logger.info(f"  - Cross-attention layers: {config.cross_attn_layers}")
    
    # Create dummy input
    batch_size = 2
    seq_length = 256
    image_size = config.image_size
    
    rng = jax.random.PRNGKey(42)
    
    input_ids = jax.random.randint(
        rng, (batch_size, seq_length), minval=0, maxval=config.vocab_size
    )
    
    images = jax.random.randint(
        rng, (batch_size, image_size, image_size, 3), minval=0, maxval=256
    ).astype(jnp.uint8)
    
    logger.info(f"\nInput shapes:")
    logger.info(f"  - Text tokens: {input_ids.shape}")
    logger.info(f"  - Images: {images.shape}")
    
    # Initialize model
    variables = model.init(
        rng,
        input_ids=input_ids,
        images=images,
        training=False
    )
    
    # Count parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(variables['params']))
    logger.info(f"  - Model parameters: {param_count:,}")
    
    # Forward pass
    logger.info("\n🔄 Running forward pass...")
    
    logits = model.apply(
        variables,
        input_ids=input_ids,
        images=images,
        training=False
    )
    
    logger.info(f"Output logits shape: {logits.shape}")
    logger.info(f"Expected shape: [batch_size={batch_size}, seq_len={seq_length}, vocab_size={config.vocab_size}]")
    
    # Test text-only mode
    logger.info("\n📝 Testing text-only mode...")
    
    text_only_logits = model.apply(
        variables,
        input_ids=input_ids,
        images=None,  # No images
        training=False
    )
    
    logger.info(f"Text-only logits shape: {text_only_logits.shape}")
    
    # Test vision encoding
    logger.info("\n🖼️ Testing vision encoder...")
    
    vision_encoder = model.vision_encoder
    vision_variables = {'params': variables['params']['vision_encoder']}
    
    vision_tokens = vision_encoder.apply(
        vision_variables,
        images=images,
        training=False
    )
    
    logger.info(f"Vision tokens shape: {vision_tokens.shape}")
    logger.info(f"Vision tokens per image: {config.num_vision_tokens}")


def demonstrate_training_pipeline():
    """Demonstrate the advanced training pipeline."""
    
    logger = setup_logging()
    logger.info("🏋️ Demonstrating Advanced Training Pipeline")
    
    # Create training configuration
    config = TrainingConfig(
        model_config=MultimodalConfig(
            embed_dim=512,  # Smaller for demo
            num_heads=8,
            num_kv_heads=4,
            vision_embed_dim=256,
            vision_layers=6,
            max_seq_len=256
        ),
        learning_rate=1e-4,
        warmup_steps=100,
        max_steps=1000,
        batch_size=4,
        use_curriculum=True,
        curriculum_stages=[
            {
                "name": "text_pretraining",
                "steps": 300,
                "modalities": ["text"],
                "learning_rate_multiplier": 1.0
            },
            {
                "name": "vision_alignment",
                "steps": 400,
                "modalities": ["text", "vision"],
                "learning_rate_multiplier": 0.5,
                "vision_weight": 2.0
            },
            {
                "name": "multimodal_finetuning",
                "steps": 300,
                "modalities": ["text", "vision"],
                "learning_rate_multiplier": 0.1,
                "cross_attention_weight": 1.5
            }
        ]
    )
    
    logger.info("Training Configuration:")
    logger.info(f"  - Model size: {config.model_config.embed_dim}D")
    logger.info(f"  - Learning rate: {config.learning_rate}")
    logger.info(f"  - Total steps: {config.max_steps}")
    logger.info(f"  - Curriculum stages: {len(config.curriculum_stages)}")
    
    # Create trainer
    trainer = MultimodalTrainer(config)
    
    # Create dummy data
    train_data = create_dummy_multimodal_data(
        batch_size=config.batch_size,
        seq_length=config.model_config.max_seq_len,
        vocab_size=config.model_config.vocab_size,
        image_size=config.model_config.image_size,
        num_batches=50
    )
    
    eval_data = create_dummy_multimodal_data(
        batch_size=config.batch_size,
        seq_length=config.model_config.max_seq_len,
        vocab_size=config.model_config.vocab_size,
        image_size=config.model_config.image_size,
        num_batches=10
    )
    
    logger.info("\n📊 Dataset Information:")
    dummy_batch = next(iter(train_data))
    logger.info(f"  - Input IDs shape: {dummy_batch['input_ids'].shape}")
    logger.info(f"  - Images shape: {dummy_batch['images'].shape}")
    logger.info(f"  - Training batches: 50")
    logger.info(f"  - Evaluation batches: 10")
    
    # Initialize training
    logger.info("\n🚀 Initializing training...")
    
    state = trainer.initialize_training(dummy_batch)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    
    logger.info(f"  - Model parameters: {param_count:,}")
    logger.info(f"  - Optimizer: {config.optimizer}")
    logger.info(f"  - Mixed precision: {config.use_mixed_precision}")
    
    # Demonstrate curriculum stages
    logger.info("\n📚 Curriculum Learning Stages:")
    
    for i, stage_config in enumerate(config.curriculum_stages):
        stage = CurriculumStage(**stage_config)
        logger.info(f"  Stage {i+1}: {stage.name}")
        logger.info(f"    - Steps: {stage.steps}")
        logger.info(f"    - Modalities: {stage.modalities}")
        logger.info(f"    - LR multiplier: {stage.learning_rate_multiplier}")
    
    # Demonstrate loss computation
    logger.info("\n💯 Loss Computation Demo:")
    
    loss, (losses, outputs) = trainer.compute_loss(
        state.params, dummy_batch, step=0, training=True
    )
    
    logger.info(f"  - Total loss: {float(loss):.4f}")
    logger.info(f"  - Loss components: {list(losses.keys())}")
    
    # Demonstrate training step
    logger.info("\n⚡ Training Step Demo:")
    
    new_state, metrics = trainer.train_step(state, dummy_batch, step=0)
    
    logger.info(f"  - Step loss: {float(metrics['total_loss']):.4f}")
    logger.info(f"  - Learning rate: {float(metrics['learning_rate']):.2e}")
    logger.info(f"  - Gradient norm: {float(metrics['grad_norm']):.4f}")
    
    logger.info("\n✨ Training pipeline ready for full training!")


def demonstrate_model_scaling():
    """Demonstrate different model sizes and configurations."""
    
    logger = setup_logging()
    logger.info("📏 Demonstrating Model Scaling")
    
    # Different model configurations
    configs = {
        "Small (1B)": MultimodalConfig(
            embed_dim=1024,
            num_heads=8,
            num_kv_heads=4,
            vision_embed_dim=512,
            vision_layers=12
        ),
        "Medium (4B)": GEMMA_4B_MULTIMODAL_CONFIG,
        "Large (12B)": GEMMA_12B_MULTIMODAL_CONFIG
    }
    
    # Create dummy input for parameter counting
    rng = jax.random.PRNGKey(42)
    dummy_input_ids = jax.random.randint(rng, (1, 256), 0, 32000)
    dummy_images = jax.random.randint(rng, (1, 224, 224, 3), 0, 256).astype(jnp.uint8)
    
    logger.info("Model Scaling Analysis:")
    logger.info("-" * 60)
    
    for name, config in configs.items():
        logger.info(f"\n{name} Parameters:")
        
        # Create model
        model = GemmaInspiredMultimodalTransformer(config=config)
        
        # Initialize to count parameters
        variables = model.init(
            rng,
            input_ids=dummy_input_ids,
            images=dummy_images,
            training=False
        )
        
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(variables['params']))
        
        logger.info(f"  - Total parameters: {param_count:,}")
        logger.info(f"  - Embedding dim: {config.embed_dim}")
        logger.info(f"  - Attention heads: {config.num_heads}")
        logger.info(f"  - KV heads: {config.num_kv_heads}")
        logger.info(f"  - Vision layers: {config.vision_layers}")
        logger.info(f"  - Vision tokens: {config.num_vision_tokens}")
        
        # Estimate memory usage (rough approximation)
        param_memory_gb = param_count * 4 / (1024**3)  # 4 bytes per float32
        training_memory_gb = param_memory_gb * 4  # Rough estimate for gradients, optimizer states
        
        logger.info(f"  - Est. param memory: {param_memory_gb:.2f} GB")
        logger.info(f"  - Est. training memory: {training_memory_gb:.2f} GB")


def main():
    """Main demonstration function."""
    
    logger = setup_logging()
    logger.info("🌟 VishwamAI Advanced Multimodal Development - Complete Demo")
    logger.info("=" * 80)
    
    print("""
    🚀 Welcome to VishwamAI Advanced Multimodal Development!
    
    This demo showcases the new formulation and techniques for building
    multimodal AI systems inspired by Google DeepMind's Gemma architecture.
    
    Key Features:
    ✅ Gemma-inspired multimodal transformer architecture
    ✅ Grouped Query Attention (GQA) for memory efficiency  
    ✅ SigLIP-style vision encoder with attention pooling
    ✅ Flash Attention 2 for training speed
    ✅ Cross-modal attention and adaptive fusion
    ✅ Advanced training pipeline with curriculum learning
    ✅ RoPE positional embeddings with scaling
    ✅ Sliding window attention for long sequences
    ✅ Multi-scale attention patterns
    ✅ Soft attention logit capping for stability
    
    """)
    
    try:
        # 1. Demonstrate attention mechanisms
        demonstrate_attention_mechanisms()
        
        print("\n" + "="*80)
        
        # 2. Demonstrate multimodal model
        demonstrate_multimodal_model()
        
        print("\n" + "="*80)
        
        # 3. Demonstrate training pipeline
        demonstrate_training_pipeline()
        
        print("\n" + "="*80)
        
        # 4. Demonstrate model scaling
        demonstrate_model_scaling()
        
        print("\n" + "="*80)
        
        logger.info("🎉 All demonstrations completed successfully!")
        logger.info("\nNext Steps:")
        logger.info("1. Integrate with your dataset using the training pipeline")
        logger.info("2. Customize model configuration for your use case")
        logger.info("3. Scale up to larger models as needed")
        logger.info("4. Experiment with different attention patterns")
        logger.info("5. Fine-tune on your specific multimodal tasks")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
