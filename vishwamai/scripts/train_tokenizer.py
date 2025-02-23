#!/usr/bin/env python3
"""Train SentencePiece tokenizer on input data."""

import argparse
import logging
from pathlib import Path
from typing import List
import yaml
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import sentencepiece as spm

from vishwamai.data.preprocessing import TextPreprocessor
from vishwamai.utils.logging import setup_logging

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to tokenizer config YAML file"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing training text files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save tokenizer model"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Vocabulary size"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000000,
        help="Number of lines to sample for training"
    )
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """Load tokenizer configuration.
    
    Args:
        config_path (str): Path to config file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def process_file(args):
    """Process a single input file.
    
    Args:
        args (tuple): Tuple containing (file_path, preprocessor)
        
    Returns:
        List[str]: List of preprocessed lines
    """
    file_path, preprocessor = args
    
    with open(file_path) as f:
        lines = f.readlines()
        
    processed_lines = []
    for line in lines:
        if line.strip():
            processed = preprocessor.preprocess(line)
            if processed:
                processed_lines.append(processed)
                
    return processed_lines

def main():
    """Main training script."""
    args = parse_args()
    setup_logging()
    
    logger.info("Loading configuration...")
    config = load_config(args.config)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(
        lowercase=config["preprocessing"]["lowercase"],
        remove_punctuation=config["preprocessing"]["remove_punctuation"],
        normalize_whitespace=config["preprocessing"]["normalize_whitespace"]
    )
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect input files
    input_files = list(input_dir.glob("**/*.txt"))
    logger.info(f"Found {len(input_files)} input files")
    
    # Process files in parallel
    all_lines = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        process_args = [(f, preprocessor) for f in input_files]
        
        for lines in tqdm(
            executor.map(process_file, process_args),
            total=len(input_files),
            desc="Processing files"
        ):
            all_lines.extend(lines)
            
    logger.info(f"Collected {len(all_lines)} lines")
    
    # Sample lines if needed
    if len(all_lines) > args.sample_size:
        import random
        all_lines = random.sample(all_lines, args.sample_size)
        logger.info(f"Sampled {args.sample_size} lines for training")
        
    # Write training data
    train_file = output_dir / "train.txt"
    with open(train_file, "w") as f:
        for line in all_lines:
            f.write(line + "\n")
            
    # Train tokenizer
    model_prefix = output_dir / "tokenizer"
    
    spm.SentencePieceTrainer.train(
        input=str(train_file),
        model_prefix=str(model_prefix),
        vocab_size=args.vocab_size,
        character_coverage=config["training"]["character_coverage"],
        model_type=config["training"]["model_type"],
        max_sentence_length=config["training"]["max_sentence_length"],
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="[PAD]",
        unk_piece="[UNK]",
        bos_piece="[BOS]",
        eos_piece="[EOS]",
        user_defined_symbols=config["training"].get("special_tokens", []),
        train_extremely_large_corpus=True,
        input_sentence_size=args.sample_size,
        shuffle_input_sentence=True,
        seed_sentencepiece_size=1000000,
        shrinking_factor=0.95,
        num_threads=args.num_workers,
        num_sub_iterations=2
    )
    
    logger.info(f"Tokenizer model saved to {model_prefix}")
    
    # Save training stats
    stats = {
        "num_input_files": len(input_files),
        "num_training_lines": len(all_lines),
        "vocab_size": args.vocab_size,
        "training_config": config["training"],
        "preprocessing_config": config["preprocessing"]
    }
    
    stats_path = output_dir / "training_stats.yaml"
    with open(stats_path, "w") as f:
        yaml.dump(stats, f)
        
if __name__ == "__main__":
    main()
