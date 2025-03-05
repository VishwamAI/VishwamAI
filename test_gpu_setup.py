#!/usr/bin/env python3
"""
Test script to verify VishwamAI setup on GPU.
"""

import os
import jax
import jax.numpy as jnp
import logging
from omegaconf import OmegaConf
from vishwamai.model import VishwamAIModel
from vishwamai.tokenizer import VishwamAITokenizer
from vishwamai.pretrain_efficient import setup_tpu_devices, create_model_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gpu_setup():
    """Test GPU setup and model initialization."""
    try:
        # Check GPU availability
        logger.info("Checking GPU setup...")
        devices = jax.devices()
        logger.info(f"Available devices: {devices}")
        
        # Load GTX 1650 optimized config
        config_path = "vishwamai/configs/training/gtx1650.yaml"
        config = OmegaConf.load(config_path)
        logger.info("Loaded configuration successfully")
        
        # Create small test batch
        batch_size = config.training.batch_size
        seq_length = config.training.curriculum.initial_length
        vocab_size = config.model.vocab_size
        
        # Create model
        logger.info("Initializing model...")
        model = VishwamAIModel(create_model_config(config))
        
        # Test forward pass with small batch
        logger.info("Testing forward pass...")
        rng = jax.random.PRNGKey(0)
        test_input = jax.random.randint(
            rng, 
            shape=(batch_size, seq_length),
            minval=0,
            maxval=vocab_size
        )
        test_mask = jnp.ones((batch_size, seq_length))
        
        # Run test forward pass
        outputs = model(test_input, attention_mask=test_mask)
        
        mem_usage = jax.device_get(outputs['logits']).nbytes / (1024 * 1024)  # MB
        logger.info(f"Forward pass successful! Memory usage: {mem_usage:.2f} MB")
        
        return {
            "status": "success",
            "memory_usage_mb": mem_usage,
            "batch_size": batch_size,
            "sequence_length": seq_length,
            "device": str(devices[0])
        }
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }

if __name__ == "__main__":
    logger.info("Starting GPU setup test...")
    result = test_gpu_setup()
    
    if result["status"] == "success":
        logger.info("=== Test Results ===")
        logger.info(f"Device: {result['device']}")
        logger.info(f"Batch Size: {result['batch_size']}")
        logger.info(f"Sequence Length: {result['sequence_length']}")
        logger.info(f"Memory Usage: {result['memory_usage_mb']:.2f} MB")
        logger.info("All tests passed successfully!")
    else:
        logger.error(f"Tests failed: {result['error']}")
