import os
import json
import logging
import shutil
from typing import List, Optional, Union, Dict, Any, Callable, Set
from pathlib import Path
import sentencepiece as spm
import numpy as np
import jax
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
        special_tokens: Optional[Dict[str, int]] = None,
        character_coverage: float = 0.99995,
        model_type: str = "bpe",
        error_tokens: bool = True,  # Whether to add error-specific tokens
    ):
        """
        Initialize the VishwamAITokenizer with the given configuration.

        Args:
            vocab_size (int): The size of the vocabulary.
            model_prefix (str): The prefix for the model files.
            pad_id (int): The ID for the padding token.
            eos_id (int): The ID for the end-of-sequence token.
            unk_id (int): The ID for the unknown token.
            bos_id (int): The ID for the beginning-of-sequence token.
            special_tokens (Dict[str, int]): Additional special tokens.
            character_coverage (float): The character coverage for training.
            model_type (str): The type of model (e.g., "bpe").
            error_tokens (bool): Whether to add error-specific tokens for error correction.
        """
        self._vocab_size = vocab_size
        self.model_prefix = model_prefix
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.unk_id = unk_id
        self.bos_id = bos_id
        self.character_coverage = character_coverage
        self.model_type = model_type
        self.sp_model = None
        
        # Initialize special tokens
        self.special_tokens = {
            "<pad>": pad_id,
            "<eos>": eos_id,
            "<unk>": unk_id,
            "<bos>": bos_id,
            "<mask>": 4,
            "<sep>": 5,
            "<cls>": 6
        }
        
        # Add custom special tokens
        if special_tokens:
            for token, idx in special_tokens.items():
                self.special_tokens[token] = idx
        
        # Add error-specific tokens if enabled
        if error_tokens:
            error_token_map = {
                "<error>": 20,              # General error indicator
                "<tot>": 21,                # Tree of Thoughts token
                "<reasoning>": 22,          # Token for reasoning
                "<correction>": 23,         # Correction indicator
                "<confidence-high>": 24,    # High confidence
                "<confidence-medium>": 25,  # Medium confidence
                "<confidence-low>": 26,     # Low confidence
                "<mod-weight>": 27,         # MoD weight indicator
                "<moe-route>": 28           # MoE routing indicator
            }
            self.special_tokens.update(error_token_map)
        
        # Create inverse map
        self.id_to_special_token = {v: k for k, v in self.special_tokens.items()}
        
        # Enhanced token cache
        self._cache = {}
        self._batch_cache = {}
        self._max_cache_size = 10000  # Limit cache size to avoid memory issues

    @property
    def vocab_size(self) -> int:
        """
        Get the vocabulary size.

        Returns:
            int: The vocabulary size.
        """
        return self.sp_model.get_piece_size() if self.sp_model else self._vocab_size
    
    @property
    def all_special_ids(self) -> List[int]:
        """Get all special token IDs."""
        return list(self.special_tokens.values())
    
    @property
    def all_special_tokens(self) -> List[str]:
        """Get all special token strings."""
        return list(self.special_tokens.keys())

    def train(self, 
             input_files: Union[str, List[str]], 
             output_dir: str,
             vocab_size: Optional[int] = None) -> None:
        """
        Train the SentencePiece tokenizer on input files.

        Args:
            input_files (Union[str, List[str]]): The input files for training.
            output_dir (str): The directory to save the trained model.
            vocab_size (Optional[int]): Override the vocabulary size.

        Raises:
            ValueError: If input files are invalid or empty.
            RuntimeError: If training fails.
        """
        logger.info("Starting tokenizer training")
        
        if not input_files:
            raise ValueError("Input files cannot be empty")
            
        if isinstance(input_files, str):
            input_files = [input_files]
            
        # Validate input files
        for file in input_files:
            if not os.path.exists(file):
                raise ValueError(f"Input file not found: {file}")
            if os.path.getsize(file) == 0:
                raise ValueError(f"Input file is empty: {file}")
            
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Use provided vocab_size if specified
        if vocab_size is not None:
            self._vocab_size = vocab_size
        
        # Extract special tokens for training
        user_defined_symbols = [
            token for token in self.special_tokens.keys()
            if token not in ["<pad>", "<eos>", "<unk>", "<bos>"]
        ]
        
        # Prepare training arguments
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
            # Train the model
            spm.SentencePieceTrainer.train(**train_args)
            
            # Load the trained model
            self.sp_model = spm.SentencePieceProcessor()
            model_path = f"{train_args['model_prefix']}.model"
            if not os.path.exists(model_path):
                raise RuntimeError("Training failed: model file not created")
                
            self.sp_model.load(model_path)
            logger.info("Tokenizer training completed")
            
        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}")

    def load(self, model_path: str) -> None:
        """
        Load a pretrained SentencePiece model.

        Args:
            model_path (str): The path to the pretrained model file.

        Raises:
            FileNotFoundError: If model file doesn't exist.
            RuntimeError: If loading fails.
        """
        logger.info(f"Loading tokenizer model from {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        try:
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.load(model_path)
            logger.info("Tokenizer model loaded")
            
            # Clear caches when loading a new model
            self._cache = {}
            self._batch_cache = {}
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def encode(self, 
              text: Union[str, List[str]], 
              add_special_tokens: bool = True,
              return_tensors: Optional[str] = None) -> Union[List[int], List[List[int]]]:
        """
        Encode text to token ids.

        Args:
            text (Union[str, List[str]]): The text to encode.
            add_special_tokens (bool): Whether to add special tokens.
            return_tensors (Optional[str]): If 'np' returns numpy array, if 'jax' returns jax array.

        Returns:
            Union[List[int], List[List[int]]]: The encoded token ids.

        Raises:
            RuntimeError: If model is not loaded or encoding fails.
        """
        if self.sp_model is None:
            raise RuntimeError("Tokenizer model not loaded. Call load() or train() first.")
            
        if isinstance(text, str):
            text = [text]
        
        # Check cache for batch encoding
        batch_key = tuple(text)
        if add_special_tokens and batch_key in self._batch_cache:
            encoded = self._batch_cache[batch_key]
            if return_tensors == 'np':
                return np.array(encoded)
            elif return_tensors == 'jax':
                return jnp.array(encoded)
            return encoded
        
        try:
            encoded = []
            for t in text:
                # Check individual cache
                if add_special_tokens and t in self._cache:
                    encoded.append(self._cache[t])
                    continue
                
                if add_special_tokens:
                    ids = [self.bos_id] + self.sp_model.encode_as_ids(t) + [self.eos_id]
                else:
                    ids = self.sp_model.encode_as_ids(t)
                
                # Update cache with LRU policy
                if add_special_tokens:
                    if len(self._cache) >= self._max_cache_size:
                        # Simple LRU - remove a random item
                        self._cache.pop(next(iter(self._cache)))
                    self._cache[t] = ids
                
                encoded.append(ids)
            
            # Update batch cache with LRU policy
            if add_special_tokens:
                if len(self._batch_cache) >= self._max_cache_size // 10:  # Smaller limit for batch cache
                    self._batch_cache.pop(next(iter(self._batch_cache)))
                self._batch_cache[batch_key] = encoded
            
            result = encoded[0] if len(encoded) == 1 else encoded
            
            # Convert to requested tensor type
            if return_tensors == 'np':
                result = np.array(result)
            elif return_tensors == 'jax':
                result = jnp.array(result)
            
            return result
        except Exception as e:
            raise RuntimeError(f"Encoding failed: {str(e)}")

    def decode(self, 
              token_ids: Union[List[int], List[List[int]], np.ndarray, jnp.ndarray], 
              skip_special_tokens: bool = True) -> Union[str, List[str]]:
        """
        Decode token ids to text.

        Args:
            token_ids (Union[List[int], List[List[int]], np.ndarray, jnp.ndarray]): Token IDs to decode.
            skip_special_tokens (bool): Whether to skip special tokens.

        Returns:
            Union[str, List[str]]: The decoded text.

        Raises:
            RuntimeError: If model is not loaded or decoding fails.
        """
        if self.sp_model is None:
            raise RuntimeError("Tokenizer model not loaded. Call load() or train() first.")
        
        # Convert numpy or jax arrays to lists
        if isinstance(token_ids, (np.ndarray, jnp.ndarray)):
            token_ids = token_ids.tolist()
            
        if isinstance(token_ids[0], (int, np.integer)):
            token_ids = [token_ids]
            
        try:
            decoded = []
            for ids in token_ids:
                if skip_special_tokens:
                    # Filter out all special token IDs
                    special_ids = set(self.all_special_ids)
                    ids = [id for id in ids if id not in special_ids]
                text = self.sp_model.decode_ids(ids)
                decoded.append(text)
                
            return decoded[0] if len(decoded) == 1 else decoded
        except Exception as e:
            raise RuntimeError(f"Decoding failed: {str(e)}")

    def add_special_tokens(self, special_tokens_dict: Dict[str, str]) -> int:
        """
        Add special tokens to the tokenizer.
        
        Args:
            special_tokens_dict: Dictionary mapping token string to token ID or just token strings.
            
        Returns:
            Number of tokens added to the vocabulary.
        """
        if self.sp_model is None:
            raise RuntimeError("Tokenizer model not loaded. Call load() or train() first.")
        
        num_added = 0
        current_vocab_size = self.vocab_size
        
        for token, token_id in special_tokens_dict.items():
            if isinstance(token_id, str):
                # If value is a string, it's just the token itself
                # Add with next available ID
                next_id = current_vocab_size + num_added
                self.special_tokens[token] = next_id
                self.id_to_special_token[next_id] = token
                num_added += 1
            else:
                # If value is an integer, use it as the token ID
                self.special_tokens[token] = token_id
                self.id_to_special_token[token_id] = token
        
        # Clear caches since vocab changed
        self._cache = {}
        self._batch_cache = {}
        
        logger.info(f"Added {num_added} special tokens")
        return num_added

    def batch_decode(self, 
                    sequences: Union[List[List[int]], np.ndarray, jnp.ndarray],
                    skip_special_tokens: bool = True,
                    clean_up_tokenization_spaces: bool = True) -> List[str]:
        """
        Decode a batch of token sequences.
        
        Args:
            sequences: List of token ID sequences
            skip_special_tokens: Whether to remove special tokens
            clean_up_tokenization_spaces: Whether to clean up spaces
            
        Returns:
            List of decoded strings
        """
        # Convert numpy or jax arrays to lists
        if isinstance(sequences, (np.ndarray, jnp.ndarray)):
            sequences = sequences.tolist()
        
        decoded_texts = self.decode(sequences, skip_special_tokens=skip_special_tokens)
        
        if clean_up_tokenization_spaces:
            decoded_texts = [self._clean_text(text) for text in decoded_texts]
            
        return decoded_texts

    def _clean_text(self, text: str) -> str:
        """Clean up a decoded text string."""
        # Remove unnecessary spaces
        text = ' '.join(text.split())
        # Fix common decoding artifacts
        text = text.replace(' .', '.')
        text = text.replace(' ,', ',')
        text = text.replace(' ?', '?')
        text = text.replace(' !', '!')
        text = text.replace(' :', ':')
        text = text.replace(' ;', ';')
        return text
    
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Convert token(s) to ID(s)."""
        if self.sp_model is None:
            raise RuntimeError("Tokenizer model not loaded. Call load() or train() first.")
        
        if isinstance(tokens, str):
            # Check special tokens first
            if tokens in self.special_tokens:
                return self.special_tokens[tokens]
            return self.sp_model.piece_to_id(tokens)
        else:
            return [self.convert_tokens_to_ids(token) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """Convert ID(s) to token(s)."""
        if self.sp_model is None:
            raise RuntimeError("Tokenizer model not loaded. Call load() or train() first.")
        
        if isinstance(ids, (int, np.integer)):
            # Check special token IDs first
            if ids in self.id_to_special_token:
                return self.id_to_special_token[ids]
            return self.sp_model.id_to_piece(ids)
        else:
            return [self.convert_ids_to_tokens(id) for id in ids]

    def token_to_error_info(self, token: Union[str, int]) -> Optional[Dict[str, Any]]:
        """
        Extract error correction information from a token.
        
        Args:
            token: Token string or ID
            
        Returns:
            Dictionary with error info or None if not an error token
        """
        # Convert ID to token string if needed
        if isinstance(token, (int, np.integer)):
            token = self.convert_ids_to_tokens(token)
            
        # Check if this is an error-related token
        error_prefixes = ["<error", "<tot", "<reasoning", "<correction", "<confidence", "<mod", "<moe"]
        
        if not any(token.startswith(prefix) for prefix in error_prefixes):
            return None
            
        info = {"is_error_token": True, "token": token}
        
        # Extract token type
        if token == "<error>":
            info["type"] = "general_error"
        elif token == "<tot>":
            info["type"] = "tree_of_thoughts"
        elif token == "<reasoning>":
            info["type"] = "reasoning"
        elif token == "<correction>":
            info["type"] = "correction"
        elif token.startswith("<confidence-"):
            info["type"] = "confidence"
            level = token[12:-1]  # Extract level from <confidence-level>
            info["level"] = level
        elif token == "<mod-weight>":
            info["type"] = "mod_weight" 
        elif token == "<moe-route>":
            info["type"] = "moe_routing"
            
        return info
        
    def get_error_tokens(self) -> Dict[str, int]:
        """Get dictionary of error-related tokens."""
        return {k: v for k, v in self.special_tokens.items() 
                if k.startswith(("<error", "<tot", "<reasoning", 
                                 "<correction", "<confidence", 
                                 "<mod", "<moe"))}

    def save(self, output_dir: str) -> None:
        """
        Save the tokenizer files.

        Args:
            output_dir (str): The directory to save the tokenizer files.

        Raises:
            RuntimeError: If model is not loaded or saving fails.
        """
        logger.info(f"Saving tokenizer to {output_dir}")
        
        if self.sp_model is None:
            raise RuntimeError("No model to save. Train or load a model first.")
            
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Copy the model file to the output directory
            source_model = f"{self.model_prefix}.model"
            source_vocab = f"{self.model_prefix}.vocab"
            target_model = os.path.join(output_dir, source_model)
            target_vocab = os.path.join(output_dir, source_vocab)
            
            if os.path.exists(source_model):
                shutil.copy2(source_model, target_model)
            if os.path.exists(source_vocab):
                shutil.copy2(source_vocab, target_vocab)
            
            # Save the configuration including special tokens
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
            
            config_path = os.path.join(output_dir, "tokenizer_config.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            logger.info("Tokenizer saved")
            
        except Exception as e:
            raise RuntimeError(f"Failed to save tokenizer: {str(e)}")

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "VishwamAITokenizer":
        """
        Load a pretrained tokenizer from directory.

        Args:
            model_dir (str): The directory containing the pretrained tokenizer.

        Returns:
            VishwamAITokenizer: The loaded tokenizer.

        Raises:
            FileNotFoundError: If required files are missing.
            RuntimeError: If loading fails.
        """
        logger.info(f"Loading pretrained tokenizer from {model_dir}")
        
        config_path = os.path.join(model_dir, "tokenizer_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        try:
            with open(config_path) as f:
                config = json.load(f)
            
            # Extract special_tokens separately to avoid constructor issues
            special_tokens = config.pop("special_tokens", None)    
            tokenizer = cls(**config)
            
            # Load the model
            model_path = os.path.join(model_dir, f"{config['model_prefix']}.model")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            tokenizer.load(model_path)
            
            # Set special tokens after loading
            if special_tokens:
                tokenizer.special_tokens = special_tokens
                tokenizer.id_to_special_token = {v: k for k, v in special_tokens.items()}
                
            logger.info("Pretrained tokenizer loaded")
            return tokenizer
            
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained tokenizer: {str(e)}")

    def get_vocab(self) -> Dict[str, int]:
        """
        Get the vocabulary mapping.

        Returns:
            Dict[str, int]: The vocabulary mapping.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self.sp_model is None:
            raise RuntimeError("Tokenizer model not loaded. Call load() or train() first.")
            
        vocab = {
            self.sp_model.id_to_piece(id): id
            for id in range(self.sp_model.get_piece_size())
        }
        
        # Add special tokens to vocab
        vocab.update(self.special_tokens)
        
        return vocab
