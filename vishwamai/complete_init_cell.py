# Import all necessary components
import torch
from vishwamai.initialize import initialize_model_and_trainer
from vishwamai.config import ModelArgs
from vishwamai.utils import precompute_freqs_cis
from vishwamai.advanced_training import AdvancedTrainer
from vishwamai.tree_of_thoughts import TreeConfig, RewardConfig
from vishwamai.curriculum import CurriculumConfig
from vishwamai.neural_memory import NeuralMemory

print("Setting up configurations...")

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
    rope_condense_ratio=1.0,
    max_steps=100000  # Set default number of training steps
)

print("\nInitializing neural memory...")
try:
    # Initialize neural memory
    neural_memory = NeuralMemory(
        args=model_args,
        memory_size=512,  # Size of memory buffer
        num_memory_heads=4  # Number of memory attention heads
    )
    print("✓ Neural memory initialized successfully")
except Exception as e:
    print(f"✗ Neural memory initialization failed: {str(e)}")
    raise

# Create training components
tot_config = TreeConfig(
    max_branches=4,
    max_depth=3,
    beam_width=2,
    reward_gamma=0.95
)

reward_config = RewardConfig(
    reasoning_weight=0.4,
    accuracy_weight=0.4,
    consistency_weight=0.2
)

curriculum_config = CurriculumConfig(
    min_sequence_length=32,
    max_sequence_length=512,
    min_vocab_complexity=0.3,
    max_vocab_complexity=1.0,
    min_reasoning_steps=1,
    max_reasoning_steps=8,
    pacing_function='root',
    total_curriculum_steps=10000,
    performance_threshold=0.8,
    min_samples_before_advance=100,
    smoothing_factor=0.95
)

# Verify model arguments
print("\nModel configuration:")
print(f"Dimension: {model_args.dim}")
print(f"Max sequence length: {model_args.max_seq_len}")
print(f"Number of layers: {model_args.n_layers}")
print(f"Number of heads: {model_args.n_heads}")
print(f"Number of routed experts: {model_args.n_routed_experts}")

print("\nVerifying frequency computation...")
try:
    # Test frequency computation
    test_freqs = precompute_freqs_cis(dim=model_args.dim, end=model_args.max_seq_len)
    print("✓ Frequency computation test successful")
except Exception as e:
    print(f"✗ Frequency computation test failed: {str(e)}")
    raise

print("\nInitializing model...")

# Execute initialization with complete arguments
try:
    model, trainer, start_step = initialize_model_and_trainer(
        model_args=model_args,
        checkpoint_dir=CHECKPOINT_DIR,
        tot_config=tot_config,
        reward_config=reward_config,
        curriculum_config=curriculum_config,
        neural_memory=neural_memory  # Pass neural memory to initialization
    )
    print(f"\nSuccessfully initialized model and trainer:")
    print(f"✓ Starting from step {start_step}")
    print(f"✓ Will train for {model_args.max_steps} steps")
    print(f"✓ Model is on device: {next(model.parameters()).device}")
    print(f"✓ Neural memory size: {neural_memory.memory_size}")
    
except Exception as e:
    print(f"\nError during initialization:")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("\nModel args state:")
    print(f"  dim: {getattr(model_args, 'dim', 'Not set')}")
    print(f"  max_seq_len: {getattr(model_args, 'max_seq_len', 'Not set')}")
    raise
