import os
import torch
import wandb
from tqdm.auto import tqdm
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TrainingLoop:
    """Manages training loop with performance tracking and checkpointing."""
    
    def __init__(
        self,
        trainer,
        model_args,
        checkpoint_dir: str,
        performance_dir: str,
        save_every: int = 1000,
        eval_every: int = 5000,
        enable_wandb: bool = True
    ):
        self.trainer = trainer
        self.model_args = model_args
        self.checkpoint_dir = Path(checkpoint_dir)
        self.performance_dir = Path(performance_dir)
        self.save_every = save_every
        self.eval_every = eval_every
        self.enable_wandb = enable_wandb
        
        # Initialize tracking
        self.performance_data = []
        self.best_eval_loss = float('inf')
        self.steps_without_improvement = 0
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.performance_dir.mkdir(parents=True, exist_ok=True)
        
    def _log_metrics(self, stats: Dict[str, Any], step: int):
        """Log metrics to wandb and local tracking."""
        if self.enable_wandb:
            wandb.log({
                "loss": stats["loss"],
                "learning_rate": stats["lr"],
                "batch_size": stats["batch_size"],
                "curriculum_level": stats["curriculum_stats"]["current_difficulty"],
                "memory_usage": stats["memory_usage"]["allocated"],
                "moe_loss": stats.get("moe_loss", 0),
                "gradient_norm": stats["gradient_norm"],
                "expert_usage": stats.get("moe_metrics", {})
            })
        
        self.performance_data.append({
            "step": step,
            **stats
        })
    
    def save_checkpoint(self, step: int, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"step_{step}.pt"
        best_path = self.checkpoint_dir / "best_model.pt"
        
        try:
            self.trainer.save_checkpoint(checkpoint_path)
            if is_best:
                self.trainer.save_checkpoint(best_path)
                
            if self.enable_wandb:
                artifact = wandb.Artifact(
                    name=f"model-checkpoint-{step}",
                    type="model"
                )
                artifact.add_file(str(checkpoint_path))
                wandb.log_artifact(artifact)
                
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            
    def train(self):
        """Execute training loop with performance tracking."""
        try:
            progress_bar = tqdm(range(self.model_args.max_steps))
            for step in progress_bar:
                # Training step
                try:
                    stats = self.trainer.train_step()
                except Exception as e:
                    logger.error(f"Error in training step: {e}")
                    continue
                    
                # Update tracking
                self._log_metrics(stats, step)
                progress_bar.set_postfix(
                    loss=f"{stats['loss']:.4f}",
                    lr=f"{stats['lr']:.2e}"
                )
                
                # Regular checkpoint
                if step > 0 and step % self.save_every == 0:
                    self.save_checkpoint(step)
                    self.plot_training_progress()
                
                # Evaluation
                if step > 0 and step % self.eval_every == 0:
                    eval_metrics = self.trainer.evaluate()
                    
                    if self.enable_wandb:
                        wandb.log({"eval": eval_metrics})
                        
                    # Check for best model
                    if eval_metrics["loss"] < self.best_eval_loss:
                        self.best_eval_loss = eval_metrics["loss"]
                        self.steps_without_improvement = 0
                        self.save_checkpoint(step, is_best=True)
                    else:
                        self.steps_without_improvement += self.eval_every
                        
                    # Early stopping check
                    if self.steps_without_improvement >= self.model_args.early_stop_patience:
                        logger.info("Early stopping triggered")
                        break
                        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.save_checkpoint(step, is_best=False)
            
        finally:
            # Save final metrics
            self.plot_training_progress()
            pd.DataFrame(self.performance_data).to_csv(
                self.performance_dir / "training_metrics.csv",
                index=False
            )
            
    def plot_training_progress(self):
        """Generate training progress visualization."""
        try:
            import matplotlib.pyplot as plt
            
            df = pd.DataFrame(self.performance_data)
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot metrics
            df.plot(x="step", y="loss", ax=axes[0,0], title="Training Loss")
            df.plot(x="step", y="learning_rate", ax=axes[0,1], title="Learning Rate")
            df.plot(x="step", y="curriculum_level", ax=axes[1,0], title="Curriculum Level")
            df.plot(x="step", y="memory_usage", ax=axes[1,1], title="Memory Usage (GB)")
            
            plt.tight_layout()
            plt.savefig(self.performance_dir / "training_progress.png")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
