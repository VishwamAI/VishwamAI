"""
Optimization utilities for training
"""
from typing import Optional, Dict, Any, Union
import torch
from torch.optim import Optimizer, AdamW, Adam, SGD
from torch.optim.lr_scheduler import LRScheduler
from torch.cuda.amp import GradScaler as TorchGradScaler
import logging

from ..config.model_config import PrecisionMode, PrecisionConfig

logger = logging.getLogger(__name__)

class GradScaler:
    """
    Advanced gradient scaling for mixed precision training
    """
    def __init__(
        self,
        precision_config: PrecisionConfig,
        init_scale: float = 2.**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: Optional[bool] = None
    ):
        self.precision_config = precision_config
        
        # Determine if scaling should be enabled based on precision mode
        if enabled is None:
            enabled = precision_config.mode in [
                PrecisionMode.FP16,
                PrecisionMode.BF16
            ] and precision_config.mixed_precision
            
        self.enabled = enabled and torch.cuda.is_available()
        
        # Create PyTorch GradScaler for FP16/BF16
        if self.enabled:
            self.scaler = TorchGradScaler(
                init_scale=precision_config.static_loss_scaling or init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval,
                enabled=True
            )
            logger.info(
                f"Initialized GradScaler for {precision_config.mode.value} "
                f"with mixed_precision={precision_config.mixed_precision}"
            )
        else:
            self.scaler = None
            logger.info(
                f"GradScaler disabled for {precision_config.mode.value} "
                f"with mixed_precision={precision_config.mixed_precision}"
            )
            
    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss based on precision mode"""
        if self.enabled:
            return self.scaler.scale(loss)
        return loss
        
    def step(self, optimizer: Optimizer) -> None:
        """Optimizer step with gradient unscaling"""
        if self.enabled:
            self.scaler.step(optimizer)
        else:
            optimizer.step()
            
    def update(self) -> None:
        """Update scale factor"""
        if self.enabled:
            self.scaler.update()
            
    def unscale_(self, optimizer: Optimizer) -> None:
        """Unscale gradients"""
        if self.enabled:
            self.scaler.unscale_(optimizer)
            
    def get_scale(self) -> float:
        """Get current scale factor"""
        if self.enabled:
            return self.scaler.get_scale()
        return 1.0
        
    def state_dict(self) -> Dict[str, Any]:
        """Get scaler state"""
        if self.enabled:
            return self.scaler.state_dict()
        return {}
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scaler state"""
        if self.enabled and state_dict:
            self.scaler.load_state_dict(state_dict)

def create_optimizer(
    model: torch.nn.Module,
    config: Dict[str, Any]
) -> Optimizer:
    """
    Create optimizer with configured parameters
    
    Args:
        model: Model to optimize
        config: Training configuration
        
    Returns:
        Configured optimizer
    """
    # Get parameter groups with weight decay config
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_params = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config["weight_decay"],
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    
    # Convert parameters to appropriate dtype based on precision mode
    if isinstance(config.get("precision"), PrecisionConfig):
        precision_mode = config["precision"].mode
        if precision_mode == PrecisionMode.FP16:
            for group in optimizer_grouped_params:
                for p in group["params"]:
                    p.data = p.data.half()
        elif precision_mode == PrecisionMode.FP64:
            for group in optimizer_grouped_params:
                for p in group["params"]:
                    p.data = p.data.double()

    # Create optimizer
    optimizer_type = config["optimizer"].lower()
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_params,
            lr=config["learning_rate"],
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif optimizer_type == "adam":
        optimizer = Adam(
            optimizer_grouped_params,
            lr=config["learning_rate"],
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif optimizer_type == "sgd":
        optimizer = SGD(
            optimizer_grouped_params,
            lr=config["learning_rate"],
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
    return optimizer

def create_scheduler(
    optimizer: Optimizer,
    config: Dict[str, Any],
    num_training_steps: int
) -> Optional[LRScheduler]:
    """
    Create learning rate scheduler
    
    Args:
        optimizer: Optimizer to schedule
        config: Training configuration
        num_training_steps: Total number of training steps
        
    Returns:
        Learning rate scheduler
    """
    # Get warmup steps
    warmup_steps = config["warmup_steps"]
    if config.get("warmup_ratio", None) is not None:
        warmup_steps = int(num_training_steps * config["warmup_ratio"])

    # Create scheduler
    scheduler_type = config["lr_scheduler"].lower()
    if scheduler_type == "linear":
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_type == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_type == "constant":
        from transformers import get_constant_schedule_with_warmup
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps
        )
    else:
        scheduler = None
        
    return scheduler
