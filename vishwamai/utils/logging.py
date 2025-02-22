"""
Logging utilities for Vishwamai
"""
import os
import sys
import logging
import json
from typing import Optional, Dict, Any
from pathlib import Path
import torch.distributed as dist

class DistributedLogger:
    """Logger that only logs on main process in distributed setting"""
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
    def should_log(self) -> bool:
        """Whether this process should log"""
        return self.rank == 0
        
    def info(self, msg: str, *args, **kwargs) -> None:
        """Log info message on main process"""
        if self.should_log():
            self.logger.info(msg, *args, **kwargs)
            
    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log warning message on main process"""
        if self.should_log():
            self.logger.warning(msg, *args, **kwargs)
            
    def error(self, msg: str, *args, **kwargs) -> None:
        """Log error message on main process"""
        if self.should_log():
            self.logger.error(msg, *args, **kwargs)
            
    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log debug message on main process"""
        if self.should_log():
            self.logger.debug(msg, *args, **kwargs)

class TrainingLogger:
    """Logger for training progress and metrics"""
    def __init__(
        self,
        output_dir: str,
        experiment_name: str = "vishwamai",
        log_level: int = logging.INFO
    ):
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logger(log_level)
        
        # Training history
        self.history = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
            "metrics": []
        }
        
    def _setup_logger(self, log_level: int) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(log_level)
        
        # Remove existing handlers
        logger.handlers = []
        
        # Create formatters and handlers
        log_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(
            self.output_dir / f"{self.experiment_name}.log"
        )
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        
        return DistributedLogger(logger)
        
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration"""
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        self.logger.info(f"Configuration saved to {config_path}")
        
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = ""
    ) -> None:
        """Log training metrics"""
        # Add metrics to history
        self.history["metrics"].append({
            "step": step,
            "metrics": metrics
        })
        
        # Format metrics string
        metrics_str = " - ".join(
            f"{prefix}{k}: {v:.4f}" for k, v in metrics.items()
        )
        self.logger.info(f"Step {step}: {metrics_str}")
        
    def log_training_step(
        self,
        loss: float,
        learning_rate: float,
        step: int,
        epoch: Optional[int] = None
    ) -> None:
        """Log training step metrics"""
        self.history["train_loss"].append(loss)
        self.history["learning_rate"].append(learning_rate)
        
        epoch_str = f"Epoch {epoch} - " if epoch is not None else ""
        self.logger.info(
            f"{epoch_str}Step {step}: loss = {loss:.4f}, lr = {learning_rate:.6f}"
        )
        
    def log_evaluation(
        self,
        metrics: Dict[str, float],
        step: int
    ) -> None:
        """Log evaluation metrics"""
        self.history["eval_loss"].append(metrics.get("eval_loss", 0.0))
        self.log_metrics(metrics, step, prefix="eval_")
        
    def save_history(self) -> None:
        """Save training history"""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        self.logger.info(f"Training history saved to {history_path}")
        
    def get_history(self) -> Dict[str, list]:
        """Get training history"""
        return self.history

def get_logger(name: str) -> logging.Logger:
    """Get logger by name"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def setup_logging(
    output_dir: str,
    experiment_name: str = "vishwamai",
    log_level: int = logging.INFO
) -> TrainingLogger:
    """Setup training logger"""
    return TrainingLogger(output_dir, experiment_name, log_level)
