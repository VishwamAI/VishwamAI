"""
Example script demonstrating training with different precision modes
"""
import os
import torch
from pathlib import Path
import logging

from vishwamai.model import VishwamaiModel
from vishwamai.config import ModelConfig, TrainingConfig, PrecisionConfig, PrecisionMode
from vishwamai.data import VishwamaiDataset, DataCollator, VishwamaiTokenizer
from vishwamai.training import Trainer
from vishwamai.utils import setup_logging

logger = logging.getLogger(__name__)

def train_with_precision(
    precision_mode: PrecisionMode,
    mixed_precision: bool = True,
    gradient_precision: str = "fp32",
    data_path: str = "data/train",
    output_dir: str = "outputs",
    batch_size: int = 32,
    num_epochs: int = 3
):
    """
    Train model with specified precision settings
    
    Args:
        precision_mode: Precision mode to use (FP16, FP32, FP64, etc.)
        mixed_precision: Whether to use mixed precision training
        gradient_precision: Precision for gradients
        data_path: Path to training data
        output_dir: Output directory
        batch_size: Training batch size
        num_epochs: Number of training epochs
    """
    # Setup output directory
    output_dir = Path(output_dir) / f"train_{precision_mode.value}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir)
    logger.info(f"Training with precision mode: {precision_mode.value}")
    
    # Create precision config
    precision_config = PrecisionConfig(
        mode=precision_mode,
        mixed_precision=mixed_precision,
        gradient_precision=gradient_precision
    )
    
    # Create model config
    model_config = ModelConfig(
        vocab_size=32000,
        hidden_size=768,
        num_layers=8,
        num_heads=12,
        intermediate_size=3072,
        precision=precision_config
    )
    
    # Create training config
    training_config = TrainingConfig(
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=1000,
        max_grad_norm=1.0
    )
    
    # Initialize model
    model = VishwamaiModel(model_config)
    
    # Setup tokenizer and dataset
    tokenizer = VishwamaiTokenizer()
    train_dataset = VishwamaiDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=1024
    )
    
    # Create data loaders
    collator = DataCollator(tokenizer=tokenizer)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        config={
            **model_config.to_dict(),
            **training_config.to_dict()
        },
        output_dir=output_dir
    )
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train()
    
    # Save final model
    trainer.save_model()
    logger.info(f"Training completed. Model saved to {output_dir}")
    
    return history

if __name__ == "__main__":
    # Example: Train with different precision modes
    
    # FP16 Mixed Precision (fastest)
    train_with_precision(
        precision_mode=PrecisionMode.FP16,
        mixed_precision=True,
        gradient_precision="fp32"
    )
    
    # FP32 Full Precision
    train_with_precision(
        precision_mode=PrecisionMode.FP32,
        mixed_precision=False,
        gradient_precision="fp32"
    )
    
    # FP64 Double Precision (slowest, highest precision)
    train_with_precision(
        precision_mode=PrecisionMode.FP64,
        mixed_precision=False,
        gradient_precision="fp64"
    )
    
    # BF16 Brain Float
    train_with_precision(
        precision_mode=PrecisionMode.BF16,
        mixed_precision=True,
        gradient_precision="fp32"
    )
