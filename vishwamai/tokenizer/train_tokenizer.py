# /home/kasinadhsarma/VishwamAI/vishwamai/tokenizer/train_tokenizer.py

import sentencepiece as spm
import logging
import os
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_tokenizer(
    input_file: str,
    model_prefix: str,
    vocab_size: int = 50000,
    model_type: str = "unigram",
    max_sentence_length: int = 512,
    additional_args: Optional[dict] = None
) -> None:
    """
    Train a SentencePiece model on the provided text corpus.

    Args:
        input_file (str): Path to the input text file (one sentence per line).
        model_prefix (str): Prefix for the output model files (e.g., 'vishwamai_tokenizer' will produce
                           'vishwamai_tokenizer.model' and 'vishwamai_tokenizer.vocab').
        vocab_size (int): Vocabulary size (default: 50000).
        model_type (str): SentencePiece model type ('unigram', 'bpe', 'char', 'word') (default: "unigram").
        max_sentence_length (int): Maximum sentence length (default: 512).
        additional_args (dict, optional): Additional arguments for SentencePiece training.
    """
    # Validate input file
    if not os.path.exists(input_file):
        logger.error(f"Input file {input_file} does not exist.")
        raise FileNotFoundError(f"Input file {input_file} does not exist.")

    # Prepare training arguments
    train_args = {
        "input": input_file,
        "model_prefix": model_prefix,
        "vocab_size": vocab_size,
        "model_type": model_type,
        "max_sentence_length": max_sentence_length,
        # Define special token IDs
        "pad_id": 0,
        "unk_id": 1,
        "bos_id": 2,
        "eos_id": 3,
        # Ensure special tokens are included in the vocabulary
        "user_defined_symbols": ",".join([
            "<pad>",
            "<think>",
            "</think>",
            "<answer>",
            "</answer>"
        ]),
        # Additional settings for efficiency
        "character_coverage": 0.9995,  # Cover 99.95% of characters
        "num_threads": os.cpu_count() or 4  # Use all available CPU cores
    }

    # Update with additional arguments if provided
    if additional_args:
        train_args.update(additional_args)

    # Convert arguments to a string for SentencePiece
    arg_str = " ".join(f"--{k}={v}" for k, v in train_args.items())
    logger.info(f"Training SentencePiece model with arguments: {arg_str}")

    # Train the model
    try:
        spm.SentencePieceTrainer.Train(arg_str)
        logger.info(f"Successfully trained SentencePiece model. Output files: {model_prefix}.model, {model_prefix}.vocab")
    except Exception as e:
        logger.error(f"Failed to train SentencePiece model: {str(e)}")
        raise

    # Verify output files
    model_file = f"{model_prefix}.model"
    vocab_file = f"{model_prefix}.vocab"
    if not (os.path.exists(model_file) and os.path.exists(vocab_file)):
        logger.error("Training completed, but output files were not created.")
        raise FileNotFoundError("Output files (.model, .vocab) were not created.")

if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer for VishwamAI.")
    parser.add_argument("input_file", type=str, help="Path to the input text file (one sentence per line).")
    parser.add_argument("model_prefix", type=str, help="Prefix for the output model files.")
    parser.add_argument("--vocab-size", type=int, default=50000, help="Vocabulary size (default: 50000).")
    parser.add_argument("--model-type", type=str, default="unigram", choices=["unigram", "bpe", "char", "word"],
                        help="SentencePiece model type (default: 'unigram').")
    parser.add_argument("--max-sentence-length", type=int, default=512, help="Maximum sentence length (default: 512).")

    args = parser.parse_args()

    # Train the tokenizer
    train_tokenizer(
        input_file=args.input_file,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        max_sentence_length=args.max_sentence_length
    )