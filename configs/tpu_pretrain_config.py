"""TPU-optimized pre-training configuration."""

def get_pretrain_config():
    """Get TPU-optimized pre-training configuration for Gemma distillation."""
    return {
        "model": {
            "vocab_size": 256000,  # Gemma vocabulary size
            "num_layers": 32,      # 7B model size
            "num_heads": 32,
            "head_dim": 128,
            "hidden_dim": 4096,
            "mlp_dim": 14336,
            "max_seq_len": 32768,
            "dropout_rate": 0.0,
            "use_flash_attn": True,
            "use_rope": True,
            "use_rms_norm": True
        },
        "thinking": {
            "num_steps": 5,         # Increased thinking steps
            "max_branches": 4,      # More thought branches
            "beam_width": 3,        # Beam search width
            "temperature": 0.7,     # Higher temperature for more diversity
            "use_value_network": True,
            "use_temporal_integration": True,
            "thought_refinement_iterations": 2
        },
        "distillation": {
            "teacher_model": "gemma-27b",
            "student_model": "gemma-7b",
            "temperature": 2.0,
            "alpha": 0.5,
            "use_intermediate_distillation": True,
            "layer_mapping_strategy": "uniform_span"
        },
        "training": {
            "batch_size": 16,
            "grad_accum_steps": 8,
            "learning_rate": 5e-5,
            "warmup_steps": 4000,
            "max_steps": 500000,
            "weight_decay": 0.1,
            "max_grad_norm": 1.0,
            "checkpoint_steps": 5000
        },
        "optimization": {
            "dtype": "bfloat16",
            "mixed_precision": True,
            "block_size": 128,
            "use_fp8_gemm": True,
            "gradient_checkpointing": True,
            "teacher_load_dtype": "bfloat16"
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