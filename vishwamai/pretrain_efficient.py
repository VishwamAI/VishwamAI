#!/usr/bin/env python3
"""
Efficient pre-training script with curriculum learning and TPU optimization.
"""

import os
import jax
import logging
from functools import partial
from omegaconf import OmegaConf
from .model import VishwamAIModel, ModelConfig
from .training import train, create_train_dataloader, create_val_dataloader
from .tokenizer import VishwamAITokenizer

logger = logging.getLogger(__name__)

def setup_tpu_devices():
    """Set up TPU devices with optimal configuration."""
    try:
        import jax.tools.colab_tpu
        jax.tools.colab_tpu.setup_tpu()
        
        # Configure bfloat16 as default for TPU
        jax.config.update('jax_default_dtype_bits', 16)
        jax.config.update('jax_default_dtype', 'bfloat16')
        
        # Create optimal device mesh for TPU
        from jax.experimental import mesh_utils
        devices = mesh_utils.create_device_mesh((8,))  # 8 TPU cores
        logger.info(f"Successfully configured TPU with {jax.device_count()} devices")
        return devices
    except Exception as e:
        logger.warning(f"Error setting up TPU: {e}. Falling back to available devices.")
        return jax.local_devices()

def create_model_config(config):
    """Create model configuration optimized for TPU."""
    model_config = ModelConfig(**config.model)
    
    # Enable TPU-specific optimizations
    model_config.use_bfloat16 = True
    model_config.gradient_checkpointing = config.training.gradient_checkpointing
    model_config.enable_pjit = config.training.enable_pjit
    
    return model_config

def main(config_path: str = "vishwamai/configs/training/efficient_pretrain.yaml"):
    """Run efficient pre-training with TPU optimizations."""
    # Load configuration
    config = OmegaConf.load(config_path)
    logger.info("Loaded configuration for efficient pre-training")
    
    # Set up TPU devices
    devices = setup_tpu_devices()
    
    # Create model
    model_config = create_model_config(config)
    model = VishwamAIModel(model_config)
    logger.info(f"Created model with {sum(p.size for p in jax.tree_leaves(model.params)):,} parameters")
    
    # Initialize tokenizer
    tokenizer = VishwamAITokenizer(vocab_size=config.model.vocab_size)
    
    # Create data loaders with curriculum learning
    train_loader = create_train_dataloader(config, tokenizer)
    val_loader = create_val_dataloader(config, tokenizer)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(config.training.checkpoint_dir, "efficient_pretrain")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Start training
    try:
        final_state = train(
            model=model,
            config=config,
            tokenizer=tokenizer,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_steps=config.training.max_steps,
            log_every=config.training.log_every_n_steps,
            eval_every=config.monitoring.save_every_n_steps,
            checkpoint_dir=checkpoint_dir,
            accum_steps=config.training.gradient_accumulation_steps,
            mesh=devices
        )
        
        # Save final model
        final_checkpoint_path = os.path.join(checkpoint_dir, "final_model")
        logger.info(f"Training completed successfully. Saving final model to {final_checkpoint_path}")
        
        return {
            "status": "success",
            "best_metrics": final_state.best_metrics,
            "final_checkpoint": final_checkpoint_path,
            "steps_completed": final_state.step
        }
        
    except Exception as e:
        logger.exception("Training failed")
        return {
            "status": "failed",
            "error": str(e)
        }

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    result = main()
    if result["status"] == "success":
        logger.info(f"Training completed successfully!")
        logger.info(f"Best metrics: {result['best_metrics']}")
        logger.info(f"Model saved to: {result['final_checkpoint']}")
    else:
        logger.error(f"Training failed: {result['error']}")
