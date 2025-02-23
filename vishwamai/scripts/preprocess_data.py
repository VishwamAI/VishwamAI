#!/usr/bin/env python3
"""Data preprocessing script for training data preparation."""

import argparse
import logging
from pathlib import Path
from typing import List, Set
import yaml
import torch
from tqdm import tqdm

from vishwamai.data.preprocessing import TextPreprocessor
from vishwamai.data.tokenization import SPTokenizer
from vishwamai.utils.logging import setup_logging

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess training data")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to data config YAML file"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing raw input files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save processed data"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Path to trained tokenizer model"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path (str): Path to config file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def main():
    """Main preprocessing script."""
    args = parse_args()
    setup_logging()
    
    logger.info("Loading configuration...")
    config = load_config(args.config)
    
    # Initialize preprocessor and tokenizer
    preprocessor = TextPreprocessor(
        lowercase=config["preprocessing"]["lowercase"],
        remove_punctuation=config["preprocessing"]["remove_punctuation"],
        normalize_whitespace=config["preprocessing"]["normalize_whitespace"]
    )
    
    tokenizer = SPTokenizer.from_pretrained(args.tokenizer_path)
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each input file
    input_files = list(input_dir.glob("**/*.txt"))
    logger.info(f"Found {len(input_files)} input files")
    
    for input_file in tqdm(input_files, desc="Processing files"):
        # Read input file
        with open(input_file) as f:
            text = f.read()
            
        # Preprocess text
        processed_text = preprocessor.preprocess(text)
        
        # Tokenize
        encodings = tokenizer.encode(
            processed_text,
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Save preprocessed and tokenized data
        relative_path = input_file.relative_to(input_dir)
        output_path = output_dir / relative_path.with_suffix(".pt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"]
        }, output_path)
        
    logger.info(f"Preprocessing complete. Processed files saved to {output_dir}")
    
    # Save preprocessing stats
    stats = {
        "num_files": len(input_files),
        "max_length": args.max_length,
        "preprocessing_config": config["preprocessing"],
        "tokenizer_path": args.tokenizer_path
    }
    
    stats_path = output_dir / "preprocessing_stats.yaml"
    with open(stats_path, "w") as f:
        yaml.dump(stats, f)
        
if __name__ == "__main__":
    main()
