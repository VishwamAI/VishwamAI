# /home/kasinadhsarma/VishwamAI/vishwamai/tokenizer/tokenizer.py

import sentencepiece as spm
import logging
from typing import List, Union, Dict, Optional
import torch

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

class DeviceAgnosticTokenizer:
    """Base class for device-agnostic tokenizer implementation"""
    def __init__(self):
        self.device_type = "gpu"  # Default to GPU
        if HAS_JAX:
            try:
                jax.devices("tpu")
                self.device_type = "tpu"
            except:
                try:
                    jax.devices("gpu")
                except:
                    pass

class SentencePieceTokenizer(DeviceAgnosticTokenizer):
    """
    A unique tokenizer for VishwamAI using SentencePiece for subword tokenization.
    Supports encoding/decoding text, handling special tokens, and integration with transformer models.
    """
    def __init__(self, model_path: str, max_seq_len: int = 512):
        """
        Initialize the SentencePieceTokenizer with a pre-trained model.

        Args:
            model_path (str): Path to the SentencePiece model file (.model).
            max_seq_len (int): Maximum sequence length for truncation (default: 512).
        """
        super().__init__()
        # Load the SentencePiece model
        try:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(model_path)
            logger.info(f"Loaded SentencePiece model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load SentencePiece model from {model_path}: {str(e)}")
            raise

        self.max_seq_len = max_seq_len
        self.vocab_size = self.sp.get_piece_size()
        
        # Define special tokens (using SentencePiece's built-in control symbols)
        self.special_tokens = {
            "pad": "<pad>",
            "unk": self.sp.unk_piece(),
            "bos": self.sp.bos_piece(),
            "eos": self.sp.eos_piece(),
            # VishwamAI-specific tokens
            "think_start": "<think>",
            "think_end": "</think>",
            "answer_start": "<answer>",
            "answer_end": "</answer>"
        }
        
        # Map special tokens to their IDs
        self.special_token_ids = {k: self.sp.piece_to_id(v) for k, v in self.special_tokens.items()}
        logger.info(f"Special tokens: {self.special_token_ids}")

    def encode(self, text: str, return_tensors: Optional[str] = None, add_special_tokens: bool = True) -> Union[List[int], torch.Tensor, jnp.ndarray]:
        """
        Encode text into token IDs with device-specific implementation.

        Args:
            text (str): Input text to encode.
            return_tensors (str, optional): "pt" for PyTorch, "jax" for JAX, None for list.
            add_special_tokens (bool): Whether to add BOS and EOS tokens.

        Returns:
            Union[List[int], torch.Tensor, jnp.ndarray]: Encoded token IDs.
        """
        # Preprocess text: ensure it's a string and not empty
        if not isinstance(text, str):
            text = str(text)
        if not text.strip():
            logger.warning("Empty text provided for encoding. Returning empty token list.")
            if return_tensors == "pt":
                return torch.tensor([], dtype=torch.long)
            elif return_tensors == "jax":
                return jnp.array([], dtype=jnp.int32)
            return []

        # Add special tokens if requested
        if add_special_tokens:
            text = f"{self.special_tokens['bos']} {text} {self.special_tokens['eos']}"

        # Encode text to token IDs
        token_ids = self.sp.encode_as_ids(text)

        # Truncate to max_seq_len if necessary (accounting for special tokens)
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len - 1] + [self.special_token_ids['eos']]
            logger.warning(f"Text truncated to max_seq_len={self.max_seq_len}")

        # Convert to tensor if requested
        if return_tensors == "pt":
            return torch.tensor(token_ids, dtype=torch.long)
        elif return_tensors == "jax":
            return jnp.array(token_ids, dtype=jnp.int32)
        return token_ids

    def decode(self, token_ids: Union[List[int], torch.Tensor, jnp.ndarray], skip_special_tokens: bool = False) -> str:
        """
        Decode token IDs back to text with device-specific handling.

        Args:
            token_ids (Union[List[int], torch.Tensor, jnp.ndarray]): Token IDs to decode.
            skip_special_tokens (bool): Whether to skip special tokens.

        Returns:
            str: Decoded text.
        """
        # Convert tensor to list if necessary
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
        elif isinstance(token_ids, jnp.ndarray):
            token_ids = token_ids.tolist()

        # Decode token IDs to text
        text = self.sp.decode_ids(token_ids)

        # Optionally remove special tokens
        if skip_special_tokens:
            for token in self.special_tokens.values():
                text = text.replace(token, "")
            text = text.strip()

        return text

    def get_vocab_size(self) -> int:
        """
        Get the vocabulary size of the tokenizer.

        Returns:
            int: Vocabulary size.
        """
        return self.vocab_size

    def get_special_tokens(self) -> Dict[str, str]:
        """
        Get the dictionary of special tokens.

        Returns:
            Dict[str, str]: Dictionary mapping special token names to their string representations.
        """
        return self.special_tokens

    def get_special_token_ids(self) -> Dict[str, int]:
        """
        Get the dictionary of special token IDs.

        Returns:
            Dict[str, int]: Dictionary mapping special token names to their IDs.
        """
        return self.special_token_ids

    def batch_encode(self, texts: List[str], return_tensors: Optional[str] = None, add_special_tokens: bool = True) -> Union[List[List[int]], torch.Tensor, jnp.ndarray]:
        """
        Encode a batch of texts with device-specific batching.

        Args:
            texts (List[str]): List of input texts to encode.
            return_tensors (str, optional): "pt" for PyTorch, "jax" for JAX, None for list.
            add_special_tokens (bool): Whether to add special tokens.

        Returns:
            Union[List[List[int]], torch.Tensor, jnp.ndarray]: Encoded token IDs.
        """
        encoded = [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]

        if return_tensors == "pt":
            # Pad sequences to the longest in the batch
            max_len = max(len(seq) for seq in encoded)
            padded = torch.full((len(encoded), max_len), self.special_token_ids['pad'], dtype=torch.long)
            for i, seq in enumerate(encoded):
                padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            return padded
        elif return_tensors == "jax":
            max_len = max(len(seq) for seq in encoded)
            padded = jnp.full((len(encoded), max_len), self.special_token_ids['pad'], dtype=jnp.int32)
            for i, seq in enumerate(encoded):
                padded = padded.at[i, :len(seq)].set(jnp.array(seq, dtype=jnp.int32))
            return padded

        return encoded

if __name__ == "__main__":
    # Example usage
    tokenizer = SentencePieceTokenizer(model_path="path/to/vishwamai_tokenizer.model")

    # Test encoding with different backends
    text = "Solve 2x + 3 = 7"
    
    # PyTorch encoding
    token_ids_pt = tokenizer.encode(text, return_tensors="pt")
    print(f"PyTorch encoded: {token_ids_pt}")
    
    if HAS_JAX:
        # JAX encoding
        token_ids_jax = tokenizer.encode(text, return_tensors="jax")
        print(f"JAX encoded: {token_ids_jax}")
    
    # Test decoding
    decoded_text = tokenizer.decode(token_ids_pt)
    print(f"Decoded: {decoded_text}")

    # Test batch encoding
    texts = ["Hello world", "Testing batch encoding"]
    batch_encoded_pt = tokenizer.batch_encode(texts, return_tensors="pt")
    print(f"Batch encoded (PyTorch): {batch_encoded_pt.shape}")
    
    if HAS_JAX:
        batch_encoded_jax = tokenizer.batch_encode(texts, return_tensors="jax")
        print(f"Batch encoded (JAX): {batch_encoded_jax.shape}")