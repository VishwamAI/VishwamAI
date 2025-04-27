"""Train a distilled model on CPU with proper error handling and logging."""

import jax
import jax.numpy as jnp
import os
import logging
import sys
import traceback
from pathlib import Path
from datetime import datetime
from absl import app, flags
from vishwamai.configs.budget_model_config import BudgetModelConfig
from vishwamai.distill import DistillationTrainer, DistillationConfig
from vishwamai.model import VishwamAI, VishwamAIConfig
from vishwamai.pipeline import DistillationDataPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/train_distill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VishwamAI-Distill")

# Force CPU execution (configurable)
os.environ["JAX_PLATFORMS"] = "cpu" 

# Create necessary directories
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Define command line flags
FLAGS = flags.FLAGS
flags.DEFINE_string("data_path", None, "Path to training data")
flags.DEFINE_integer("max_steps", None, "Override default training steps")
flags.DEFINE_float("temperature", 2.0, "Distillation temperature")
flags.DEFINE_float("alpha", 0.5, "Distillation loss weight")
flags.DEFINE_string("checkpoint_dir", "checkpoints/distill", "Directory to save checkpoints")
flags.DEFINE_boolean("resume", False, "Resume training from checkpoint if available")
flags.DEFINE_integer("seed", 42, "Random seed for reproducibility")

def count_parameters(params):
    """Count the total number of parameters in a model."""
    return sum(p.size for p in jax.tree_util.tree_leaves(params))

def main(_):
    try:
        # Set random seed for reproducibility
        jax.random.PRNGKey(FLAGS.seed)
        
        logger.info("Initializing budget configuration")
        config = BudgetModelConfig()
        
        # Create teacher model (larger model)
        logger.info("Creating teacher model (large model)")
        teacher_config = VishwamAIConfig(
            vocab_size=32000,
            hidden_dim=2048,
            num_layers=24,
            num_heads=16,
            head_dim=128,
            mlp_dim=8192,
            max_seq_len=2048
        )
        teacher_model = VishwamAI(config=teacher_config)
        
        # Create student model (smaller model based on budget config)
        logger.info("Creating student model (distilled/budget model)")
        student_config = VishwamAIConfig(
            vocab_size=config.model_config["vocab_size"],
            hidden_dim=config.model_config["hidden_size"],
            num_layers=config.model_config["num_hidden_layers"],
            num_heads=config.model_config["num_attention_heads"],
            head_dim=config.model_config["hidden_size"] // config.model_config["num_attention_heads"],
            mlp_dim=config.model_config["intermediate_size"],
            max_seq_len=config.model_config["max_position_embeddings"]
        )
        student_model = VishwamAI(config=student_config)
        
        # Initialize distillation trainer with proper error handling
        logger.info(f"Initializing distillation trainer with temperature={FLAGS.temperature}, alpha={FLAGS.alpha}")
        try:
            trainer = DistillationTrainer(
                teacher_model=teacher_model,
                student_model=student_model,
                student_config=student_config,
                temperature=FLAGS.temperature,
                alpha=FLAGS.alpha,
                checkpoint_dir=FLAGS.checkpoint_dir
            )
        except Exception as e:
            logger.error(f"Error initializing distillation trainer: {e}")
            logger.error(traceback.format_exc())
            sys.exit(1)
        
        # Create data pipeline with proper error handling
        logger.info("Creating distillation data pipeline")
        try:
            pipeline = DistillationDataPipeline(
                config=config,
                teacher_model=teacher_model,
                devices=jax.devices(),
                enable_thinking=True
            )
        except Exception as e:
            logger.error(f"Error creating data pipeline: {e}")
            logger.error(traceback.format_exc())
            sys.exit(1)
        
        # Check if data path exists
        if not os.path.exists(FLAGS.data_path):
            logger.error(f"Data path {FLAGS.data_path} does not exist")
            sys.exit(1)
            
        # Create training dataset
        logger.info(f"Creating distillation dataset from {FLAGS.data_path}")
        try:
            train_dataset = pipeline.create_distillation_dataset(
                FLAGS.data_path,
                is_training=True,
                cache_teacher_outputs=True
            )
        except Exception as e:
            logger.error(f"Error creating training dataset: {e}")
            logger.error(traceback.format_exc())
            sys.exit(1)
        
        # Remove duplicate print statements
        # Calculate model sizes
        teacher_params_count = count_parameters(teacher_model.params)
        student_params_count = count_parameters(student_model.params)
        reduction_ratio = student_params_count / teacher_params_count
        
        logger.info("Starting distillation training...")
        logger.info(f"Teacher model size: {teacher_params_count:,} parameters")
        logger.info(f"Student model size: {student_params_count:,} parameters")
        logger.info(f"Reduction factor: {reduction_ratio:.2%}")
        logger.info(f"Training for {FLAGS.max_steps or config.training_config['max_steps']} steps")
        
        # Prepare training configuration
        train_config = {
            "learning_rate": config.training_config["learning_rate"],
            "batch_size": config.training_config["batch_size"],
            "gradient_accumulation_steps": config.training_config["gradient_accumulation_steps"],
            "use_gradient_checkpointing": config.memory_config["use_gradient_checkpointing"]
        }
        
        logger.info(f"Training configuration: {train_config}")
        
        # Train with distillation and handle errors
        try:
            trainer.train(
                train_dataset=train_dataset,
                max_steps=FLAGS.max_steps or config.training_config["max_steps"],
                eval_steps=config.training_config["eval_steps"],
                save_steps=config.training_config["save_steps"],
                resume=FLAGS.resume,
                **train_config
            )
            
            logger.info("Training complete!")
            
            # Save final student model
            final_model_path = os.path.join(FLAGS.checkpoint_dir, "final_model")
            logger.info(f"Saving final student model to {final_model_path}")
            trainer.save_student_model(final_model_path)
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user. Saving checkpoint...")
            trainer.save_checkpoint("interrupted_checkpoint")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            logger.error(traceback.format_exc())
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error during setup: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    # Validate required flags
    flags.mark_flag_as_required("data_path")
    
    try:
        app.run(main)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
