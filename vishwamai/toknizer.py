"""
Conceptual Tokenizer Implementation
=================================

This module implements a tokenizer that understands mathematical and logical concepts.
"""

from typing import List, Optional, Dict, Union
from dataclasses import dataclass
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import json
import os

@dataclass
class ConceptualTokenizerConfig:
    """Configuration for the ConceptualTokenizer."""
    
    vocab_size: int = 32000
    max_length: int = 4096
    model_type: str = "unigram"  # "unigram", "bpe", "wordpiece"
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    mask_token: str = "<mask>"
    concept_tokens: Optional[List[str]] = None
    reasoning_tokens: Optional[List[str]] = None
    
    def __post_init__(self):
        self.special_tokens = {
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "mask_token": self.mask_token
        }
        
        # Initialize concept and reasoning tokens if not provided
        if self.concept_tokens is None:
            self.concept_tokens = []
        if self.reasoning_tokens is None:
            self.reasoning_tokens = []

class ConceptualTokenizer:
    """
    A tokenizer that understands mathematical concepts and logical reasoning patterns.
    Built on top of HuggingFace's tokenizer with additional features for handling
    mathematical notation and logical structures.
    """
    
    def __init__(self, config: ConceptualTokenizerConfig):
        self.config = config
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        
        # Initialize vocabulary with special tokens
        self._initialize_special_tokens()
        
        # Add concept and reasoning tokens
        self._add_special_tokens(config.concept_tokens)
        self._add_special_tokens(config.reasoning_tokens)
        
        # Initialize tokenizer backend based on model type
        self._initialize_backend()
        
    def _initialize_special_tokens(self):
        """Initialize vocabulary with special tokens."""
        for token in self.config.special_tokens.values():
            if token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.inverse_vocab[idx] = token
                
    def _add_special_tokens(self, tokens: List[str]):
        """Add special tokens to vocabulary."""
        for token in tokens:
            if token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.inverse_vocab[idx] = token
                
    def _initialize_backend(self):
        """Initialize the tokenizer backend based on model type."""
        if self.config.model_type == "unigram":
            from transformers import T5TokenizerFast
            self.backend = T5TokenizerFast.from_pretrained(
                "t5-small", 
                model_max_length=self.config.max_length
            )
        elif self.config.model_type == "bpe":
            from transformers import GPT2TokenizerFast
            self.backend = GPT2TokenizerFast.from_pretrained(
                "gpt2",
                model_max_length=self.config.max_length
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
        # Add custom tokens to backend
        special_tokens = {
            "pad_token": self.config.pad_token,
            "unk_token": self.config.unk_token,
            "bos_token": self.config.bos_token,
            "eos_token": self.config.eos_token,
            "mask_token": self.config.mask_token,
            "additional_special_tokens": self.config.concept_tokens + self.config.reasoning_tokens
        }
        self.backend.add_special_tokens(special_tokens)
        
    def encode(
        self, 
        text: Union[str, List[str]], 
        return_tensors: Optional[str] = None,
        padding: bool = False,
        truncation: bool = True,
        max_length: Optional[int] = None
    ) -> Union[List[int], torch.Tensor]:
        """Encode text into token IDs."""
        if max_length is None:
            max_length = self.config.max_length
            
        encoded = self.backend.encode(
            text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors
        )
        
        return encoded
        
    def decode(
        self, 
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True
    ) -> str:
        """Decode token IDs back to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        return self.backend.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )
        
    def save_pretrained(self, save_directory: str):
        """Save tokenizer configuration and vocabulary."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save config
        config_dict = {
            "vocab_size": self.config.vocab_size,
            "max_length": self.config.max_length,
            "model_type": self.config.model_type,
            "special_tokens": self.config.special_tokens,
            "concept_tokens": self.config.concept_tokens,
            "reasoning_tokens": self.config.reasoning_tokens
        }
        
        with open(os.path.join(save_directory, "tokenizer_config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
            
        # Save vocabulary
        self.backend.save_pretrained(save_directory)
        
    @classmethod
    def from_pretrained(cls, pretrained_path: str) -> "ConceptualTokenizer":
        """Load a pretrained tokenizer."""
        # Load config
        with open(os.path.join(pretrained_path, "tokenizer_config.json"), "r") as f:
            config_dict = json.load(f)
            
        config = ConceptualTokenizerConfig(**config_dict)
        tokenizer = cls(config)
        
        # Load backend
        if config.model_type == "unigram":
            from transformers import T5TokenizerFast
            tokenizer.backend = T5TokenizerFast.from_pretrained(pretrained_path)
        elif config.model_type == "bpe":
            from transformers import GPT2TokenizerFast
            tokenizer.backend = GPT2TokenizerFast.from_pretrained(pretrained_path)
            
        return tokenizer
