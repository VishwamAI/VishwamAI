#!/usr/bin/env python3
"""
VishwamAI main entry point.
Handles model training, evaluation, and inference coordination.
"""

import os
import argparse
import logging
from typing import Optional

from vishwamai.models.transformer import VishwamAITransformer
from vishwamai.models.cot_model import CoTModel
from vishwamai.models.tot_model import ToTModel
from vishwamai.training.train_normal import Trainer
from vishwamai.training.train_cot import CoTTrainer
from vishwamai.training.train_tot import ToTTrainer
from vishwamai.training.dataset_loader import VishwamAIDataset
from vishwamai.tokenizer.tokenizer import SentencePieceTokenizer
from vishwamai.configs.integration_config import IntegrationConfig

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO"):
    """Configure logging"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_tokenizer(tokenizer_path: str) -> SentencePieceTokenizer:
    """Load the tokenizer from path"""
    return SentencePieceTokenizer(
        model_path=tokenizer_path,
        max_seq_len=512
    )

def get_trainer_class(training_mode: str):
    """Get appropriate trainer class based on mode"""
    return {
        "normal": Trainer,
        "cot": CoTTrainer,
        "tot": ToTTrainer
    }.get(training_mode, Trainer)

def get_model_class(model_type: str):
    """Get appropriate model class based on type"""
    return {
        "normal": VishwamAITransformer,
        "cot": CoTModel,
        "tot": ToTModel
    }.get(model_type, VishwamAITransformer)

def main(args):
    """Main execution function"""
    setup_logging(args.log_level)
    logger.info("Initializing VishwamAI")

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer_path)
    logger.info(f"Loaded tokenizer with vocab size {tokenizer.vocab_size}")

    # Initialize model
    Model = get_model_class(args.model_type)
    model = Model(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim
    )
    logger.info(f"Initialized {args.model_type} model")

    # Load datasets
    train_dataset = VishwamAIDataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
        mode=args.model_type
    )
    
    val_dataset = VishwamAIDataset(
        data_path=args.val_data,
        tokenizer=tokenizer,
        mode=args.model_type
    ) if args.val_data else None

    # Initialize trainer
    Trainer = get_trainer_class(args.model_type)
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config={
            'batch_size': args.batch_size,
            'grad_acc_steps': args.grad_acc_steps,
            'learning_rate': args.learning_rate,
            'warmup_steps': args.warmup_steps,
            'max_steps': args.max_steps,
            'fp16': args.fp16,
            'local_rank': int(os.environ.get('LOCAL_RANK', -1)),
            'experiment_name': args.experiment_name
        }
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VishwamAI Training")
    
    # Model configuration
    parser.add_argument("--model-type", type=str, default="normal",
                       choices=["normal", "cot", "tot"],
                       help="Model type to train")
    parser.add_argument("--embed-dim", type=int, default=768,
                       help="Embedding dimension")
    parser.add_argument("--num-layers", type=int, default=12,
                       help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=12,
                       help="Number of attention heads")
    parser.add_argument("--ff-dim", type=int, default=3072,
                       help="Feed-forward dimension")

    # Training configuration
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Training batch size")
    parser.add_argument("--grad-acc-steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                       help="Learning rate warmup steps")
    parser.add_argument("--max-steps", type=int, default=50000,
                       help="Maximum training steps")
    parser.add_argument("--fp16", action="store_true",
                       help="Use mixed precision training")

    # Data configuration
    parser.add_argument("--train-data", type=str, required=True,
                       help="Path to training data")
    parser.add_argument("--val-data", type=str,
                       help="Path to validation data")
    parser.add_argument("--tokenizer-path", type=str, required=True,
                       help="Path to tokenizer model")

    # Logging configuration
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--experiment-name", type=str, default="vishwamai_training",
                       help="Name for experiment tracking")

    args = parser.parse_args()
    main(args)