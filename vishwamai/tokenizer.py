# tokenizer.py
import os
import json
from typing import List, Optional, Union, Dict
from pathlib import Path
import sentencepiece as spm
import re
from collections import Counter
import logging
import numpy as np
import jax.numpy as jnp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VishwamAITokenizer:
    def __init__(
        self,
        vocab_size: int = 32000,
        model_prefix: str = "vishwamai",
        pad_id: int = 0,
        eos_id: int = 1,
        unk_id: int = 2,
        bos_id: int = 3,
        character_coverage: float = 0.99995,
        model_type: str = "bpe",
    ):
        """Initialize the tokenizer with adaptive settings."""
        self._vocab_size = vocab_size
        self.model_prefix = model_prefix
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.unk_id = unk_id
        self.bos_id = bos_id
        self.character_coverage = character_coverage
        self.model_type = model_type
        self.sp_model = None

        # Default special tokens
        self.special_tokens = {
            "<pad>": pad_id,
            "<eos>": eos_id,
            "<unk>": unk_id,
            "<bos>": bos_id,
            "<mask>": 4,
            "<sep>": 5,
            "<cls>": 6,
            "<number>": 7,  # Added for datasets like GSM8K
            "<operator>": 8,
            "<equals>": 9
        }
        self.id_to_special_token = {v: k for k, v in self.special_tokens.items()}
        self._cache = {}
        self._batch_cache = {}
        self._max_cache_size = 10000

    def train(self, input_files: Union[str, List[str]], output_dir: str, vocab_size: Optional[int] = None) -> None:
        """Train the tokenizer, adapting to dataset-specific patterns."""
        logger.info("Starting tokenizer training")
        
        if not input_files:
            raise ValueError("Input files cannot be empty")
        if isinstance(input_files, str):
            input_files = [input_files]
        for file in input_files:
            if not os.path.exists(file) or os.path.getsize(file) == 0:
                raise ValueError(f"Invalid input file: {file}")

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if vocab_size is not None:
            self._vocab_size = vocab_size

        # Analyze dataset for frequent patterns
        dataset_analysis = self._analyze_dataset(input_files)
        user_defined_symbols = list(self.special_tokens.keys()) + dataset_analysis["frequent_patterns"]
        self.character_coverage = dataset_analysis["optimal_character_coverage"]

        train_args = {
            "input": ",".join(input_files),
            "model_prefix": os.path.join(output_dir, self.model_prefix),
            "vocab_size": self._vocab_size,
            "character_coverage": self.character_coverage,
            "pad_id": self.pad_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
            "bos_id": self.bos_id,
            "model_type": self.model_type,
            "user_defined_symbols": user_defined_symbols,
            "control_symbols": ["<start>", "<end>"],
            "normalization_rule_name": "nmt_nfkc_cf"
        }

        try:
            spm.SentencePieceTrainer.train(**train_args)
            self.sp_model = spm.SentencePieceProcessor()
            model_path = f"{train_args['model_prefix']}.model"
            if not os.path.exists(model_path):
                raise RuntimeError("Training failed: model file not created")
            self.sp_model.load(model_path)
            logger.info("Tokenizer training completed")
        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}")

    def _analyze_dataset(self, input_files: List[str]) -> Dict[str, Union[float, List[str]]]:
        """Analyze dataset to adapt tokenizer settings."""
        all_text = ""
        for file in input_files:
            with open(file, "r", encoding="utf-8") as f:
                all_text += f.read()

        # Identify frequent patterns (e.g., numbers, operators)
        patterns = re.findall(r'\d+|[+\-*/=()]', all_text)
        frequent_patterns = [pattern for pattern, count in Counter(patterns).most_common(5)]

        # Adjust character_coverage based on special character ratio
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', all_text)) / len(all_text)
        optimal_character_coverage = 0.99995 if special_char_ratio < 0.1 else 0.995

        return {
            "frequent_patterns": frequent_patterns,
            "optimal_character_coverage": optimal_character_coverage
        }

    def preprocess_text(self, text: str) -> str:
        """Preprocess text by tagging numbers and operators."""
        text = re.sub(r'\d+', '<number>', text)
        text = re.sub(r'[+\-*/=]', '<operator>', text)
        return text

    def encode(self, text: Union[str, List[str]], add_special_tokens: bool = True) -> Union[List[int], List[List[int]]]:
        """Encode text into token IDs with preprocessing."""
        if self.sp_model is None:
            raise RuntimeError("Tokenizer model not loaded. Call train() first.")

        if isinstance(text, str):
            text = [self.preprocess_text(text)]
        else:
            text = [self.preprocess_text(t) for t in text]

        batch_key = tuple(text)
        if add_special_tokens and batch_key in self._batch_cache:
            return self._batch_cache[batch_key]

        try:
            encoded = []
            for t in text:
                if add_special_tokens and t in self._cache:
                    encoded.append(self._cache[t])
                    continue
                ids = ([self.bos_id] + self.sp_model.encode_as_ids(t) + [self.eos_id]) if add_special_tokens else self.sp_model.encode_as_ids(t)
                if add_special_tokens:
                    if len(self._cache) < self._max_cache_size:
                        self._cache[t] = ids
                encoded.append(ids)

            if add_special_tokens:
                if len(self._batch_cache) < self._max_cache_size // 10:
                    self._batch_cache[batch_key] = encoded

            return encoded[0] if len(encoded) == 1 else encoded
        except Exception as e:
            raise RuntimeError(f"Encoding failed: {str(e)}")

    def decode(self, token_ids: Union[List[int], List[List[int]], np.ndarray, jnp.ndarray]) -> Union[str, List[str]]:
        """Decode token IDs back to text."""
        if self.sp_model is None:
            raise RuntimeError("Tokenizer model not loaded. Call train() first.")
        
        if isinstance(token_ids, (np.ndarray, jnp.ndarray)):
            token_ids = token_ids.tolist()
            
        if isinstance(token_ids[0], (int, np.integer)):
            token_ids = [token_ids]
            
        decoded = []
        for ids in token_ids:
            special_ids = set(self.special_tokens.values())
            ids = [id for id in ids if id not in special_ids]  # Skip special tokens
            text = self.sp_model.decode_ids(ids)
            decoded.append(text)
                
        return decoded[0] if len(decoded) == 1 else decoded

    def save(self, output_dir: str) -> None:
        """Save the tokenizer model and config."""
        if self.sp_model is None:
            raise RuntimeError("No model to save. Train a model first.")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model_path = os.path.join(output_dir, f"{self.model_prefix}.model")
        self.sp_model.serialized_model_proto()  # Save model
        with open(model_path, 'wb') as f:
            f.write(self.sp_model.serialized_model_proto())
        config = {
            "vocab_size": self._vocab_size,
            "model_prefix": self.model_prefix,
            "pad_id": self.pad_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
            "bos_id": self.bos_id,
            "character_coverage": self.character_coverage,
            "model_type": self.model_type,
            "special_tokens": self.special_tokens
        }
        with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "VishwamAITokenizer":
        """Load a pretrained tokenizer."""
        config_path = os.path.join(model_dir, "tokenizer_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path) as f:
            config = json.load(f)
        tokenizer = cls(**{k: v for k, v in config.items() if k != "special_tokens"})
        model_path = os.path.join(model_dir, f"{config['model_prefix']}.model")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        tokenizer.sp_model = spm.SentencePieceProcessor()
        tokenizer.sp_model.load(model_path)
        tokenizer.special_tokens = config["special_tokens"]
        tokenizer.id_to_special_token = {v: k for k, v in config["special_tokens"].items()}
        return tokenizer