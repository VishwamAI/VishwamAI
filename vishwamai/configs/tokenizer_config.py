# VishwamAI/configs/tokenizer_config.py
"""
Settings for configuring and training the VishwamAI tokenizer.
Supports both SentencePiece and Hugging Face tokenizers with device optimizations.
"""

import os
import logging
from typing import Optional, Dict, Any

try:
    import jax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

logger = logging.getLogger(__name__)

class TokenizerConfig:
    # General tokenizer settings
    TOKENIZER_TYPE = "sentencepiece"  # Options: "sentencepiece", "huggingface"
    VOCAB_SIZE = 50000  # Base vocabulary size
    MAX_LENGTH = 512  # Maximum sequence length
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    MASK_TOKEN = "[MASK]"
    THINK_TOKEN = "<think>"
    THINK_END_TOKEN = "</think>"
    ANSWER_TOKEN = "<answer>"
    ANSWER_END_TOKEN = "</answer>"

    # Device-specific settings
    TPU_OPTIMIZATION = HAS_JAX  # Enable TPU-specific optimizations if JAX is available
    USE_FAST_TOKENIZER = True  # Use faster tokenizer implementation when available
    ENABLE_CACHING = True  # Enable tokenizer caching for faster processing
    BATCH_SIZE = 1024  # Batch size for tokenizer training
    NUM_WORKERS = 4  # Number of workers for parallel processing

    # Training settings
    TRAIN_CUSTOM = True  # Train custom tokenizer instead of using pretrained
    TRAINING_DATA_PATH = "data/train_corpus.txt"
    MIN_FREQUENCY = 2  # Minimum token frequency
    SAMPLING_FACTOR = 1e-4  # Subword regularization sampling factor
    CHARACTER_COVERAGE = 0.9995
    SPLIT_DIGITS = True
    ADD_DUMMY_PREFIX = True
    NORMALIZATION_RULE_NAME = "nmt_nfkc"  # Normalization rule

    # Model paths
    MODEL_DIR = "tokenizers"
    MODEL_PREFIX = "vishwamai_tokenizer"
    PRETRAINED_PATH = None  # Path to pretrained tokenizer if not training custom

    @classmethod
    def get_model_path(cls) -> str:
        """Get the full path to the tokenizer model"""
        return os.path.join(cls.MODEL_DIR, f"{cls.MODEL_PREFIX}.model")

    @classmethod
    def get_special_tokens(cls) -> Dict[str, str]:
        """Get mapping of special tokens"""
        return {
            "pad_token": cls.PAD_TOKEN,
            "unk_token": cls.UNK_TOKEN,
            "bos_token": cls.BOS_TOKEN,
            "eos_token": cls.EOS_TOKEN,
            "mask_token": cls.MASK_TOKEN,
            "think_token": cls.THINK_TOKEN,
            "think_end_token": cls.THINK_END_TOKEN,
            "answer_token": cls.ANSWER_TOKEN,
            "answer_end_token": cls.ANSWER_END_TOKEN
        }

    @classmethod
    def get_training_args(cls) -> Dict[str, Any]:
        """Get tokenizer training arguments"""
        return {
            "vocab_size": cls.VOCAB_SIZE,
            "character_coverage": cls.CHARACTER_COVERAGE,
            "model_type": "unigram",
            "split_digits": cls.SPLIT_DIGITS,
            "add_dummy_prefix": cls.ADD_DUMMY_PREFIX,
            "normalization_rule_name": cls.NORMALIZATION_RULE_NAME,
            "sampling_factor": cls.SAMPLING_FACTOR,
            "num_threads": cls.NUM_WORKERS,
            "pad_id": 0,
            "unk_id": 1,
            "bos_id": 2,
            "eos_id": 3,
            "user_defined_symbols": ",".join([
                cls.MASK_TOKEN,
                cls.THINK_TOKEN,
                cls.THINK_END_TOKEN,
                cls.ANSWER_TOKEN,
                cls.ANSWER_END_TOKEN
            ])
        }

    @classmethod
    def load_tokenizer(cls, model_path: Optional[str] = None):
        """Load or configure the tokenizer based on settings"""
        from vishwamai.tokenizer.tokenizer import SentencePieceTokenizer

        if model_path is None:
            model_path = cls.get_model_path()

        if not os.path.exists(model_path) and cls.PRETRAINED_PATH:
            model_path = cls.PRETRAINED_PATH

        if not os.path.exists(model_path):
            raise ValueError(
                f"No tokenizer found at {model_path}. "
                "Train a new tokenizer first using train_tokenizer.py"
            )

        return SentencePieceTokenizer(
            model_path=model_path,
            max_seq_len=cls.MAX_LENGTH
        )

if __name__ == "__main__":
    # Test the configuration
    tokenizer = TokenizerConfig.load_tokenizer()
    print("Tokenizer Loaded:", tokenizer)
    print("Vocab Size:", len(tokenizer))