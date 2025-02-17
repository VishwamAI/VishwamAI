import os
import torch
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from dataclasses import asdict
from .config import ModelArgs
from .advanced_training import AdvancedTrainer
from .tree_of_thoughts import TreeConfig
from .curriculum import CurriculumConfig
import wandb
from .model_factory import create_model
from .neural_memory import NeuralMemory

# Define default paths
DEFAULT_CHECKPOINT_DIR = os.path.join(os.path.expanduser('~'), 'VishwamAI', 'checkpoints')
CHECKPOINT_DIR = os.getenv('VISHWAMAI_CHECKPOINT_DIR', DEFAULT_CHECKPOINT_DIR)

def initialize_model_and_trainer(
    model_args: ModelArgs,
    checkpoint_dir: Optional[str] = None,
    tot_config: Optional[TreeConfig] = None,
    reward_config: Optional[Dict] = None,
    curriculum_config: Optional[CurriculumConfig] = None,
    neural_memory: NeuralMemory = None
) -> Tuple[torch.nn.Module, AdvancedTrainer, int]:
    """
    Initialize model and trainer with proper configuration and checkpoint handling.
    """
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR
    
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Validating model arguments...")
    print(f"Dimension: {model_args.dim}")
    print(f"Max sequence length: {model_args.max_seq_len}")
    
    # Validate model_args
    if not hasattr(model_args, 'dim') or not hasattr(model_args, 'max_seq_len'):
        raise ValueError("model_args must have 'dim' and 'max_seq_len' attributes")
    
    # Create neural memory if not provided
    if neural_memory is None:
        print("Initializing default neural memory...")
        neural_memory = NeuralMemory(
            args=model_args,
            memory_size=512,
            num_memory_heads=4
        )
    
    # Check for latest checkpoint
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob("step_*.pt"))
    latest_checkpoint = None
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[1]))
        print(f"Found checkpoint: {latest_checkpoint}")

    try:
        # Create model - this will handle precision settings internally
        model, tokenizer = create_model(model_args, device=device)
        
        # Load from checkpoint if available
        if latest_checkpoint:
            print("Restoring from checkpoint...")
            checkpoint = torch.load(latest_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_step = checkpoint.get('step', 0)
            if 'neural_memory_state' in checkpoint:
                neural_memory.load_state_dict(checkpoint['neural_memory_state'])
        else:
            print("Starting fresh training...")
            start_step = 0

        # Move neural memory to device
        neural_memory = neural_memory.to(device)

        # Initialize trainer with config dictionary
        config_dict = asdict(model_args)
        trainer = AdvancedTrainer(
            model=model,
            config=config_dict,
            device=device,
            memory_size=512,
            cache_size=256,
            tot_config=tot_config,
            reward_config=reward_config,
            curriculum_config=curriculum_config,
            neural_memory=neural_memory
        )

        if latest_checkpoint:
            trainer.load_state_dict(checkpoint['trainer_state_dict'])

        # Initialize or resume wandb
        run_id = os.getenv('WANDB_RUN_ID')
        if run_id and latest_checkpoint:
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
                    "curriculum": curriculum_config.__dict__ if curriculum_config else {},
                    "tot": tot_config.__dict__ if tot_config else {},
                    "memory": {
                        "size": neural_memory.memory_size,
                        "num_heads": neural_memory.num_memory_heads
                    }
                }
            )
            os.environ['WANDB_RUN_ID'] = wandb.run.id

        print(f"Model initialized on {device}")
        print(f"Memory usage: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")
        print(f"Starting from step: {start_step}")
        print(f"Neural memory size: {neural_memory.memory_size}")

        return model, trainer, start_step

    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        raise

def get_checkpoint_dir() -> str:
    """Get the checkpoint directory, creating it if necessary."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return CHECKPOINT_DIR
