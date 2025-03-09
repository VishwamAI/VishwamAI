"""
Device-agnostic tokenizer implementation for VishwamAI.
Optimized for both TPU and GPU training.
"""

import sentencepiece as spm
import logging
from typing import List, Union, Dict, Optional
import torch

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

logger = logging.getLogger(__name__)

class DeviceAgnosticTokenizer:
    """Base class for device-agnostic tokenizer implementation"""
    def __init__(self, max_seq_len: int = 512):
        self.max_seq_len = max_seq_len
        self.special_token_ids = {
            'pad': 0,
            'unk': 1,
            'bos': 2,
            'eos': 3,
            'mask': 4,
            'think': 5,
            'think_end': 6,
            'answer': 7,
            'answer_end': 8
        }
        self._reverse_special_tokens = {v: k for k, v in self.special_token_ids.items()}
        
    @property
    def vocab_size(self) -> int:
        raise NotImplementedError
        
    def encode(self, text: str, return_tensors: Optional[str] = None) -> Union[List[int], torch.Tensor, "jnp.ndarray"]:
        raise NotImplementedError
        
    def decode(self, token_ids: Union[List[int], torch.Tensor, "jnp.ndarray"]) -> str:
        raise NotImplementedError

class SentencePieceTokenizer(DeviceAgnosticTokenizer):
    """SentencePiece tokenizer with optimizations for TPU/GPU"""
    def __init__(self, model_path: str, max_seq_len: int = 512):
        super().__init__(max_seq_len)
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self._vocab_size = self.sp.GetPieceSize()

        # Pre-compile JAX functions if available
        if HAS_JAX:
            self._jax_pad = jax.jit(self._pad_sequence)
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def _pad_sequence(self, seq: List[int], max_len: int, pad_id: int = 0) -> List[int]:
        """Pad sequence to given length"""
        return seq + [pad_id] * (max_len - len(seq))

    def encode(self, text: str, return_tensors: Optional[str] = None, add_special_tokens: bool = True) -> Union[List[int], torch.Tensor, "jnp.ndarray"]:
        """
        Encode text to token ids with optional tensor conversion.
        Optimized for batch processing on TPU/GPU.
        """
        if isinstance(text, str):
            token_ids = self.sp.EncodeAsIds(text)
            if add_special_tokens:
                token_ids = [self.special_token_ids['bos']] + token_ids + [self.special_token_ids['eos']]
            token_ids = token_ids[:self.max_seq_len]
        else:
            raise ValueError("Input must be string")

        if return_tensors == "pt":
            return torch.tensor(token_ids, dtype=torch.long)
        elif return_tensors == "jax" and HAS_JAX:
            return jnp.array(token_ids, dtype=jnp.int32)
        return token_ids

    def batch_encode(self, texts: List[str], return_tensors: Optional[str] = None, add_special_tokens: bool = True) -> Union[List[List[int]], torch.Tensor, "jnp.ndarray"]:
        """Optimized batch encoding"""
        encoded = [
            self.encode(text, return_tensors=None, add_special_tokens=add_special_tokens)
            for text in texts
        ]
        
        if return_tensors is None:
            return encoded
            
        # Get max length for padding
        max_len = min(max(len(seq) for seq in encoded), self.max_seq_len)
        
        # Pad sequences
        padded = [self._pad_sequence(seq, max_len) for seq in encoded]
            
        if return_tensors == "pt":
            return torch.tensor(padded, dtype=torch.long)
        elif return_tensors == "jax" and HAS_JAX:
            return jnp.array(padded, dtype=jnp.int32)
        return padded

    def decode(self, token_ids: Union[List[int], torch.Tensor, "jnp.ndarray"], skip_special_tokens: bool = False) -> str:
        """
        Decode token ids back to text.
        Handles both PyTorch and JAX tensors efficiently.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
        elif HAS_JAX and isinstance(token_ids, jnp.ndarray):
            token_ids = token_ids.tolist()
        elif not isinstance(token_ids, list):
            token_ids = [token_ids]
            
        # Filter special tokens if requested
        if skip_special_tokens:
            token_ids = [t for t in token_ids if t not in self._reverse_special_tokens]
            
        return self.sp.DecodeIds(token_ids)

    def add_special_tokens(self, special_tokens: Dict[str, str]) -> int:
        """Add new special tokens to the tokenizer"""
        num_added = 0
        for token_id, token in special_tokens.items():
            if token not in self.special_token_ids:
                self.special_token_ids[token] = self._vocab_size + num_added
                self._reverse_special_tokens[self._vocab_size + num_added] = token
                num_added += 1
        return num_added

if __name__ == "__main__":
    # Example usage and testing
    tokenizer = SentencePieceTokenizer(model_path="path/to/model.model")
    
    # Test encoding with different backends
    text = "Testing the tokenizer"
    token_ids_pt = tokenizer.encode(text, return_tensors="pt")
    print("PyTorch encoded:", token_ids_pt)
    
    if HAS_JAX:
        token_ids_jax = tokenizer.encode(text, return_tensors="jax")
        print("JAX encoded:", token_ids_jax)
    
    # Test batch encoding
    texts = ["First sequence", "Second sequence"]
    batch_encoded = tokenizer.batch_encode(texts, return_tensors="pt")
    print("Batch encoded shape:", batch_encoded.shape)
    
    # Test decoding
    decoded = tokenizer.decode(token_ids_pt)
    print("Decoded:", decoded)