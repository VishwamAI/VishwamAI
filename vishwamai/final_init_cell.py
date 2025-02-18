# Initialize model and trainer
from vishwamai.initialize import initialize_model_and_trainer

print("Starting model initialization...")
print(f"Using model dimensions: {model_args.dim}")
print(f"Using sequence length: {model_args.max_seq_len}")

# Add max_steps to model_args if not present
if not hasattr(model_args, 'max_steps'):
    setattr(model_args, 'max_steps', 100000)  # Set default max steps

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
