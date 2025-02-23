#!/usr/bin/env python3
"""Evaluation script for Vishwamai model."""

import argparse
import logging
import json
from pathlib import Path
import yaml
import torch
import torch_xla.core.xla_model as xm
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass

from vishwamai.data.tokenization import SPTokenizer
from vishwamai.data.dataset.implementations import (
    MMBenchDataset, MMBenchMetrics,
    MMUDataset, MMUMetrics,
    GSM8KDataset, GSM8KMetrics
)
from vishwamai.model.transformer.model import VishwamaiModel
from vishwamai.training.distributed import TPUManager, setup_tpu
from vishwamai.utils.logging import setup_logging

logger = logging.getLogger(__name__)

@dataclass
class EvalConfig:
    """Evaluation configuration."""
    model_path: str
    model_config_path: str
    tokenizer_path: str
    data_dir: str
    output_dir: str
    batch_size: int
    max_length: int
    benchmark: str
    num_samples: int

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Vishwamai model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to model config YAML"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Path to tokenizer model"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing evaluation data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["mmlu", "mmmu", "gsm8k"],
        required=True,
        help="Benchmark to evaluate on"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None for all)"
    )
    return parser.parse_args()

def load_model(model_path: str, model_config: dict, device) -> VishwamaiModel:
    """Load trained model.
    
    Args:
        model_path (str): Path to model checkpoint
        model_config (dict): Model configuration
        device: Device to load model on
        
    Returns:
        VishwamaiModel: Loaded model
    """
    model = VishwamaiModel(model_config)
    checkpoint = torch.load(model_path, map_location=device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    return model

def evaluate(config: EvalConfig):
    """Main evaluation function.
    
    Args:
        config (EvalConfig): Evaluation configuration
    """
    # Setup TPU
    setup_tpu()
    tpu_manager = TPUManager()
    
    # Load model configuration
    with open(config.model_config_path) as f:
        model_config = yaml.safe_load(f)
        
    # Create tokenizer and model
    tokenizer = SPTokenizer.from_pretrained(config.tokenizer_path)
    model = load_model(config.model_path, model_config, tpu_manager.device)
    model.eval()
    
    # Create dataset and metrics
    if config.benchmark == "mmlu":
        dataset = MMBenchDataset(
            data_dir=config.data_dir,
            tokenizer=tokenizer,
            max_length=config.max_length
        )
        metrics = MMBenchMetrics()
    elif config.benchmark == "mmmu":
        dataset = MMUDataset(
            data_dir=config.data_dir,
            tokenizer=tokenizer,
            max_length=config.max_length
        )
        metrics = MMUMetrics()
    else:  # gsm8k
        dataset = GSM8KDataset(
            data_dir=config.data_dir,
            tokenizer=tokenizer,
            max_length=config.max_length
        )
        metrics = GSM8KMetrics()
        
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    dataloader = tpu_manager.get_parallel_loader(dataloader)
    
    # Evaluation loop
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
            
            if config.num_samples and len(all_predictions) >= config.num_samples:
                all_predictions = all_predictions[:config.num_samples]
                all_labels = all_labels[:config.num_samples]
                break
                
    # Calculate metrics
    results = metrics.compute(
        predictions=np.array(all_predictions),
        labels=np.array(all_labels)
    )
    
    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / f"{config.benchmark}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
        
    # Log results
    logger.info(f"\nEvaluation Results ({config.benchmark}):")
    for metric, value in results.items():
        logger.info(f"{metric}: {value:.4f}")
        
def main():
    """Main evaluation script."""
    args = parse_args()
    setup_logging()
    
    config = EvalConfig(
        model_path=args.model_path,
        model_config_path=args.model_config,
        tokenizer_path=args.tokenizer_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        benchmark=args.benchmark,
        num_samples=args.num_samples
    )
    
    evaluate(config)
    
if __name__ == "__main__":
    main()
