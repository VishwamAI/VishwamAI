# /home/kasinadhsarma/VishwamAI/vishwamai/tokenizer/train_tokenizer.py

import sentencepiece as spm
import logging
import os
from typing import Optional, Dict, Any
import torch
import multiprocessing

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    import flax.linen as nn
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_device_type():
    """Determine optimal device for training."""
    if HAS_JAX:
        try:
            jax.devices("tpu")
            return "tpu"
        except:
            try:
                jax.devices("gpu")
                return "gpu"
            except:
                pass
    return "gpu" if torch.cuda.is_available() else "cpu"

def optimize_training_params(device_type: str, vocab_size: int) -> Dict[str, Any]:
    """Get optimized training parameters based on device."""
    base_params = {
        "vocab_size": vocab_size,
        "character_coverage": 0.9995,
        "num_threads": multiprocessing.cpu_count()
    }
    
    if device_type == "tpu":
        # TPU-optimized parameters
        base_params.update({
            "model_type": "unigram",  # Unigram works better with TPU
            "train_extremely_large_corpus": "true",
            "shuffle_on_disk": "true",
            "num_sub_iterations": 2,
            "optimization_level": 3
        })
    elif device_type == "gpu":
        # GPU-optimized parameters
        base_params.update({
            "model_type": "bpe",  # BPE tends to be faster on GPU
            "enable_differential_privacy": "false",
            "required_chars": "",
            "max_sentence_length": 8192,
            "num_threads": min(multiprocessing.cpu_count() * 2, 32)  # More aggressive threading for GPU
        })
    
    return base_params

def train_tokenizer(
    input_file: str,
    model_prefix: str,
    vocab_size: int = 50000,
    model_type: str = "unigram",
    max_sentence_length: int = 512,
    additional_args: Optional[dict] = None,
    force_device: Optional[str] = None
) -> None:
    """
    Train a SentencePiece model with device-specific optimizations.

    Args:
        input_file (str): Path to the input text file (one sentence per line).
        model_prefix (str): Prefix for the output model files.
        vocab_size (int): Vocabulary size (default: 50000).
        model_type (str): SentencePiece model type (default: "unigram").
        max_sentence_length (int): Maximum sentence length (default: 512).
        additional_args (dict, optional): Additional arguments for training.
        force_device (str, optional): Force specific device type ("tpu", "gpu", or "cpu").
    """
    if not os.path.exists(input_file):
        logger.error(f"Input file {input_file} does not exist.")
        raise FileNotFoundError(f"Input file {input_file} does not exist.")

    # Determine device type
    device_type = force_device or get_device_type()
    logger.info(f"Training on device type: {device_type}")

    # Get optimized parameters for the device
    train_args = optimize_training_params(device_type, vocab_size)
    
    # Update with base parameters
    train_args.update({
        "input": input_file,
        "model_prefix": model_prefix,
        "model_type": model_type,
        "max_sentence_length": max_sentence_length,
        "pad_id": 0,
        "unk_id": 1,
        "bos_id": 2,
        "eos_id": 3,
        "user_defined_symbols": ",".join([
            "<pad>",
            "<think>",
            "</think>",
            "<answer>",
            "</answer>"
        ])
    })

    # Update with any additional arguments
    if additional_args:
        train_args.update(additional_args)

    # Convert arguments to string format
    arg_str = " ".join(f"--{k}={v}" for k, v in train_args.items())
    logger.info(f"Training SentencePiece model with device-specific optimizations. Arguments: {arg_str}")

    try:
        if device_type == "tpu":
            # TPU-specific preprocessing
            logger.info("Performing TPU-specific optimizations...")
            with jax.devices("tpu")[0]:
                spm.SentencePieceTrainer.Train(arg_str)
        else:
            # GPU/CPU training
            if device_type == "gpu" and torch.cuda.is_available():
                logger.info("Performing GPU-specific optimizations...")
                torch.cuda.empty_cache()  # Clear GPU cache
            
            spm.SentencePieceTrainer.Train(arg_str)

        logger.info(f"Successfully trained model. Output files: {model_prefix}.model, {model_prefix}.vocab")
    except Exception as e:
        logger.error(f"Failed to train model: {str(e)}")
        raise

    # Verify outputs
    model_file = f"{model_prefix}.model"
    vocab_file = f"{model_prefix}.vocab"
    if not (os.path.exists(model_file) and os.path.exists(vocab_file)):
        logger.error("Training completed but output files were not created.")
        raise FileNotFoundError("Output files not created.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a device-optimized SentencePiece tokenizer for VishwamAI.")
    parser.add_argument("input_file", type=str, help="Path to the input text file.")
    parser.add_argument("model_prefix", type=str, help="Prefix for the output model files.")
    parser.add_argument("--vocab-size", type=int, default=50000, help="Vocabulary size (default: 50000).")
    parser.add_argument("--model-type", type=str, default="unigram", 
                       choices=["unigram", "bpe", "char", "word"],
                       help="SentencePiece model type (default: 'unigram').")
    parser.add_argument("--max-sentence-length", type=int, default=512,
                       help="Maximum sentence length (default: 512).")
    parser.add_argument("--device", type=str, choices=["tpu", "gpu", "cpu"],
                       help="Force specific device type.")

    args = parser.parse_args()

    train_tokenizer(
        input_file=args.input_file,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        max_sentence_length=args.max_sentence_length,
        force_device=args.device
    )