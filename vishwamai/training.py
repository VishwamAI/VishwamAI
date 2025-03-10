"""
Training module for VishwamAI transformer.
"""

import jax
import jax.numpy as jnp
import optax
import time
from typing import Any, Dict, Optional, Tuple, Callable, Iterator
from tqdm.auto import tqdm
from .pipeline import VishwamAIPipeline
from .transformer import create_learning_rate_schedule
from .logger import DuckDBLogger

class VishwamAITrainer:
    """Training manager for VishwamAI models."""
    
    def __init__(
        self,
        pipeline: VishwamAIPipeline,
        config: Dict[str, Any],
        train_loader: Iterator,
        eval_loader: Optional[Iterator] = None,
        experiment_name: Optional[str] = None,
        db_path: str = "training_logs.db"
    ):
        self.pipeline = pipeline
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        
        # Training state
        self.current_step = 0
        self.best_eval_loss = float('inf')
        self.early_stopping_counter = 0
        
        # Initialize learning rate schedule
        self.lr_schedule = create_learning_rate_schedule(
            base_learning_rate=config['training']['learning_rate'],
            warmup_steps=config['training']['warmup_steps'],
            decay_steps=config['training']['train_steps']
        )
        
        # Setup training
        self.pipeline.setup_training(self.lr_schedule)
        
        # Initialize DuckDB logger
        self.logger = DuckDBLogger(
            db_path=db_path,
            experiment_name=experiment_name,
            config=config
        )
    
    def train(self):
        """Run the training loop."""
        print("Starting training...")
        total_steps = self.config['training']['train_steps']
        
        try:
            for step in tqdm(range(self.current_step, total_steps)):
                self.current_step = step
                
                # Get next batch
                try:
                    batch = next(self.train_loader)
                except StopIteration:
                    print("Reached end of dataset, restarting...")
                    continue
                
                # Training step
                metrics = self._train_step(batch)
                
                # Log metrics
                self._log_metrics(metrics, step)
                
                # Evaluate and save checkpoint if needed
                if step % self.config['logging']['eval_every'] == 0:
                    self._run_evaluation(step)
                
                # Save checkpoint
                if step % self.config['logging']['save_every'] == 0:
                    self._save_checkpoint(step)
                
                # Check for early stopping
                if self._should_stop():
                    print("Early stopping triggered")
                    break
                    
        finally:
            # Export final logs and close logger
            self._export_final_logs()
            self.logger.close()
    
    def _train_step(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        """Execute single training step."""
        dropout_rng = jax.random.PRNGKey(int(time.time()))
        
        # Perform training step
        new_state, metrics = self.pipeline.train_step(
            self.pipeline.state,
            batch,
            dropout_rng
        )
        
        # Update state
        self.pipeline.state = new_state
        
        return metrics
    
    def _run_evaluation(self, step: int):
        """Run evaluation loop."""
        if self.eval_loader is None:
            return
            
        eval_metrics = []
        for _ in range(self.config.get('eval_steps', 100)):
            try:
                batch = next(self.eval_loader)
            except StopIteration:
                break
                
            # Get evaluation metrics
            metrics = self.pipeline.eval_step(
                self.pipeline.state,
                batch
            )
            eval_metrics.append(metrics)
        
        # Compute average metrics
        avg_metrics = {
            k: jnp.mean([m[k] for m in eval_metrics])
            for k in eval_metrics[0].keys()
        }
        
        # Log evaluation metrics
        self._log_metrics(avg_metrics, step, prefix='eval')
        
        # Update best loss and early stopping
        if avg_metrics['loss'] < self.best_eval_loss:
            self.best_eval_loss = avg_metrics['loss']
            self.early_stopping_counter = 0
            
            # Save best model
            self._save_checkpoint(step, is_best=True)
        else:
            self.early_stopping_counter += 1
    
    def _log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = 'train'
    ):
        """Log metrics using DuckDB logger."""
        # Format metrics for logging
        log_metrics = {
            k: float(v)
            for k, v in metrics.items()
        }
        
        # Log to DuckDB
        self.logger.log_metrics(log_metrics, step, prefix)
        
        # Print to console
        if step % self.config['logging']['log_every'] == 0:
            metrics_str = " ".join(
                f"{k}: {v:.4f}"
                for k, v in log_metrics.items()
            )
            print(f"Step {step}: {metrics_str}")
    
    def _save_checkpoint(self, step: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = self.config['checkpoint_dir']
        
        # Save current checkpoint
        checkpoint_path = f"{checkpoint_dir}/step_{step}.ckpt"
        self.pipeline.save_checkpoint(checkpoint_path)
        
        # Save best model if needed
        if is_best:
            best_path = f"{checkpoint_dir}/best_model.ckpt"
            self.pipeline.save_checkpoint(best_path)
    
    def _should_stop(self) -> bool:
        """Check if training should stop early."""
        patience = self.config.get('early_stopping_patience', 5)
        return self.early_stopping_counter >= patience
    
    def _export_final_logs(self):
        """Export final training logs and summary."""
        # Export logs to CSV
        self.logger.export_to_csv()
        
        # Get and print experiment summary
        summary = self.logger.get_experiment_summary()
        print("\nTraining Summary:")
        print(f"Total Steps: {summary['total_steps']}")
        print(f"Duration: {summary['end_time'] - summary['start_time']}")
        print("\nMetrics Summary:")
        for metric, stats in summary['metrics_summary'].items():
            print(f"{metric}:")
            print(f"  Min: {stats['min']:.4f}")
            print(f"  Max: {stats['max']:.4f}")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std: {stats['std']:.4f}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        self.pipeline.load_checkpoint(path)
    
def create_trainer(
    config: Dict[str, Any],
    train_loader: Iterator,
    eval_loader: Optional[Iterator] = None,
    tokenizer: Optional[Any] = None,
    model: Optional[Any] = None,
    teacher_model: Optional[Any] = None,
    experiment_name: Optional[str] = None,
    db_path: str = "training_logs.db"
) -> VishwamAITrainer:
    """Create a trainer instance with all components."""
    
    # Create pipeline
    pipeline = VishwamAIPipeline(
        config=config,
        tokenizer=tokenizer,
        model=model,
        teacher_model=teacher_model
    )
    
    # Create and return trainer
    return VishwamAITrainer(
        pipeline=pipeline,
        config=config,
        train_loader=train_loader,
        eval_loader=eval_loader,
        experiment_name=experiment_name,
        db_path=db_path
    )

