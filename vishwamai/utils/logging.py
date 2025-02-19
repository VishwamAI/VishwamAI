import logging
import wandb
from typing import Dict, Any
from datetime import datetime
import yaml
import os

class PretrainingLogger:
    def __init__(self, config_path: str):
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        # Initialize wandb
        if self.config['experiment']['tracking']['platform'] == 'wandb':
            wandb.init(
                project=self.config['experiment']['tracking']['project'],
                entity=self.config['experiment']['tracking']['entity'],
                config=self.config
            )
            
        # Setup file logging
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        self.logger = logging.getLogger('VishwamAI_Pretrain')
        self.logger.setLevel(logging.INFO)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fh = logging.FileHandler(f'{log_dir}/pretrain_{timestamp}.log')
        fh.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log training/evaluation metrics."""
        # Log to wandb
        wandb.log(metrics, step=step)
        
        # Log to file
        metric_str = ' | '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step}: {metric_str}")
        
    def log_hardware_stats(self, gpu_stats: Dict[str, Any]):
        """Log hardware utilization metrics."""
        self.logger.info(f"GPU Stats: {gpu_stats}")
        wandb.log(gpu_stats)
        
    def log_checkpoint(self, checkpoint_path: str, epoch: int):
        """Log checkpoint saving."""
        self.logger.info(f"Saved checkpoint at epoch {epoch}: {checkpoint_path}")
        wandb.save(checkpoint_path)
        
    def log_error(self, error: Exception):
        """Log errors during training."""
        self.logger.error(f"Error occurred: {str(error)}", exc_info=True)
        
    def finish(self):
        """Cleanup logging."""
        wandb.finish()
