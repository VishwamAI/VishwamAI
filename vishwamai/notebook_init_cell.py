# Initialize model and trainer with restart handling
from vishwamai.initialize import initialize_model_and_trainer

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
except Exception as e:
    print(f"Error during initialization: {str(e)}")
    raise
