import torch
import os
from pathlib import Path
import wandb
from .model_factory import create_model
from .advanced_training import AdvancedTrainer
from .fp8_cast_bf16 import main

def initialize_model_and_trainer(model_args, checkpoint_dir, tot_config, reward_config, curriculum_config):
    """Initialize model and trainer with proper configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check for latest checkpoint
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob("step_*.pt"))
    latest_checkpoint = None
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[1]))
        print(f"Found checkpoint: {latest_checkpoint}")

    try:
        if (latest_checkpoint):
            # Load from checkpoint
            print("Restoring from checkpoint...")
            checkpoint = torch.load(latest_checkpoint)
            model, _ = create_model(model_args)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_step = checkpoint.get('step', 0)
        else:
            # Fresh start
            print("Starting fresh training...")
            model, _ = create_model(model_args)
            start_step = 0

        model = model.to(device)
        main(model)  # Apply FP8/BF16 optimizations

        # Initialize trainer
        trainer = AdvancedTrainer(
            model=model,
            config=model_args.__dict__,  # Convert ModelArgs to dict for trainer
            device=device,
            memory_size=512,
            cache_size=256,
            tot_config=tot_config,
            reward_config=reward_config,
            curriculum_config=curriculum_config
        )

        if (latest_checkpoint):
            trainer.load_state_dict(checkpoint['trainer_state_dict'])

        # Initialize or resume wandb
        run_id = os.getenv('WANDB_RUN_ID')
        if (run_id and latest_checkpoint):
            wandb.init(
                project="vishwamai-training",
                id=run_id,
                resume="must"
            )
        else:
            wandb.init(
                project="vishwamai-training",
                config={
                    "model": model_args.__dict__,
                    "curriculum": curriculum_config.__dict__,
                    "tot": tot_config.__dict__
                }
            )
            os.environ['WANDB_RUN_ID'] = wandb.run.id

        print(f"Model initialized on {device}")
        print(f"Memory usage: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")
        print(f"Starting from step: {start_step}")

        return model, trainer, start_step

    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        raise
