"""TPU-optimized pre-training configuration."""

def get_pretrain_config():
    """Get TPU-optimized pre-training configuration."""
    return {
        "model": {
            "vocab_size": 32000,
            "num_layers": 24,
            "num_heads": 16,
            "head_dim": 64,
            "hidden_dim": 1024,  # Multiple of 128 for TPU efficiency
            "mlp_dim": 4096,
            "max_seq_len": 2048,
            "dropout_rate": 0.1,
            "use_flash_attn": True,
            "use_rotary": True,
            "use_rms_norm": True
        },
        "training": {
            "batch_size": 16,  # Per device batch size
            "grad_accum_steps": 4,  # Global batch size = 16 * 4 * 8 = 512
            "learning_rate": 1e-4,
            "warmup_steps": 2000,
            "max_steps": 100000,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
        },
        "optimization": {
            "dtype": "bfloat16",  # TPU native format
            "mixed_precision": True,
            "block_size": 128,  # Optimal for TPU
        },
        "tpu": {
            "num_devices": 8,
            "device_strategy": "data_parallel",
            "mesh_shape": {
                "data": 8,
                "model": 1
            }
        }
    }