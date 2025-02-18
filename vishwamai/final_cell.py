# Initialize model and trainer
from vishwamai.initialize import initialize_model_and_trainer
from vishwamai.config import ModelArgs

# Create model arguments with explicit typing
model_args = ModelArgs(
    max_batch_size=4,
    max_seq_len=2048,
    dtype="fp8",
    vocab_size=32000,
    dim=1024,
    inter_dim=2816,
    moe_inter_dim=512,
    n_layers=12,
    n_dense_layers=1,
    n_heads=16,
    n_routed_experts=8,
    n_shared_experts=1,
    n_activated_experts=2,
    n_expert_groups=1,
    n_limited_groups=1,
    score_func="softmax",
    route_scale=1.0,
    q_lora_rank=0,
    kv_lora_rank=64,
    qk_nope_head_dim=64,
    qk_rope_head_dim=32,
    v_head_dim=64,
    original_seq_len=2048,
    rope_theta=10000.0,
    rope_factor=20,
    beta_fast=16,
    beta_slow=1,
    mscale=0.5,
    use_alibi=False,
    use_rope_scaling=True,
    gradient_checkpointing=True,
    parallel_attn=True,
    rope_condense_ratio=1.0
)

print("Starting model initialization...")
print(f"Using model dimensions: {model_args.dim}")
print(f"Using sequence length: {model_args.max_seq_len}")

# Execute initialization with complete arguments
try:
    model, trainer, start_step = initialize_model_and_trainer(
        model_args=model_args,
        checkpoint_dir=CHECKPOINT_DIR,
        tot_config=tot_config,
        reward_config=reward_config,
        curriculum_config=curriculum_config
    )
    print(f"Successfully initialized model and trainer at step {start_step}")
    print(f"Model will train for {model_args.max_steps} steps")
except Exception as e:
    print(f"Error during initialization: {str(e)}")
    raise
