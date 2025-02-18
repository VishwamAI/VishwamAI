# Initialize model and trainer
from vishwamai.initialize import initialize_model_and_trainer

# Import required modules
import os
from pathlib import Path
import wandb
import torch

# Execute initialization with all required arguments
model, trainer, start_step = initialize_model_and_trainer(
    model_args=model_args,
    checkpoint_dir=CHECKPOINT_DIR,
    tot_config=tot_config,
    reward_config=reward_config,
    curriculum_config=curriculum_config
)

# Add max_steps to model_args for training loop
if not hasattr(model_args, 'max_steps'):
    model_args.max_steps = 100000  # Set default max steps

print(f"Model initialized successfully. Starting from step {start_step}")
