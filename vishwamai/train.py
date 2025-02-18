import os
import logging
import torch
from typing import Optional, Tuple

from .gpu_utils import GPUManager
from .config import ModelArgs
from .initialize import initialize_model_and_trainer
from .tree_of_thoughts import TreeConfig
from .curriculum import CurriculumConfig
from .reward_function import RewardConfig

logger = logging.getLogger(__name__)

def initialize_training(
    model_args: ModelArgs,
    checkpoint_dir: str,
    tot_config: TreeConfig,
    reward_config: RewardConfig,
    curriculum_config: CurriculumConfig
) -> Tuple[torch.nn.Module, object, int]:
    """Initialize training environment and model."""
    try:
        # Setup GPU
        gpu_manager = GPUManager()
        gpu_manager.optimize_settings()
        
        # Get model configuration
        gpu_manager.adjust_model_config(model_args)
        
        # Initialize model with optimized settings
        model, trainer, start_step = initialize_model_and_trainer(
            model_args=model_args,
            checkpoint_dir=checkpoint_dir,
            tot_config=tot_config,
            reward_config=reward_config,
            curriculum_config=curriculum_config
        )
        
        # Apply memory optimizations
        if hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()
        elif hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            
        if hasattr(model, "enable_mem_efficient_attention"):
            model.enable_mem_efficient_attention()
            
        logger.info("Model initialization successful!")
        logger.info(f"Starting training from step {start_step}")
        logger.info(f"Will train for {model_args.max_steps} steps")
        
        return model, trainer, start_step
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error("GPU out of memory!")
        logger.error(f"Available memory: {gpu_manager.spec.memory_total - gpu_manager.spec.memory_used:.1f} GB")
        logger.error("Try reducing model size or batch size")
        raise e
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        raise

def main():
    """Main training entry point."""
    try:
        # Define configurations
        model_args = ModelArgs()  # Add your model args here
        tot_config = TreeConfig()  # Add your tree config here
        reward_config = RewardConfig()  # Add your reward config here  
        curriculum_config = CurriculumConfig()  # Add your curriculum config here
        
        checkpoint_dir = os.getenv('CHECKPOINT_DIR', './checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Execute initialization
        model, trainer, start_step = initialize_training(
            model_args=model_args,
            checkpoint_dir=checkpoint_dir,
            tot_config=tot_config,
            reward_config=reward_config,
            curriculum_config=curriculum_config
        )
        
        # Initialize training loop
        training_loop = TrainingLoop(
            trainer=trainer,
            model_args=model_args,
            checkpoint_dir=checkpoint_dir,
            performance_dir=DRIVE_DIR,
            save_every=1000,
            eval_every=5000
        )
        
        # Execute training
        training_loop.train()
        
        print("Training complete!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
        
if __name__ == "__main__":
    main()
