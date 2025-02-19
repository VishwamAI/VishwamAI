"""
SentencePiece-based tokenizer implementation for VishwamAI models.

This module provides tokenization functionality using SentencePiece for
subword tokenization, with fallback to transformers tokenizers.
"""

import json
import os
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import torch
import numpy as np

try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False

try:
    from transformers import PreTrainedTokenizer, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class Tokenizer:
    """
    Tokenizer using SentencePiece for subword tokenization.
    """
    
    def __init__(
        self,
        vocab_file: Optional[Union[str, Path]] = None,
        spm_file: Optional[Union[str, Path]] = None,
        pretrained_model_name: Optional[str] = None,
        max_length: int = 2048,
        add_special_tokens: bool = True
    ):
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        
        # Load pretrained tokenizer if specified
        if pretrained_model_name is not None:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "transformers package is required to use pretrained tokenizers. "
                    "Install with: pip install transformers"
                )
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
            self._using_transformers = True
            return
            
        # Load SentencePiece model if provided
        if spm_file is not None:
            if not SENTENCEPIECE_AVAILABLE:
                raise ImportError(
                    "sentencepiece package is required. "
                    "Install with: pip install sentencepiece"
                )
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(str(spm_file))
            self._using_sentencepiece = True
            
            # Load vocabulary
            self.vocab = {
                self.sp.IdToPiece(i): i
                for i in range(self.sp.GetPieceSize())
            }
            self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
            
        # Load custom vocabulary if provided
        elif vocab_file is not None:
            self.vocab = self._load_vocab(vocab_file)
            self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
            self._using_sentencepiece = False
            self._using_transformers = False
            
        else:
            raise ValueError(
                "Either pretrained_model_name, spm_file, or vocab_file must be provided"
            )
            
        # Special tokens
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"
        self.mask_token = "[MASK]"
        
        # Add special tokens to vocabulary if using SentencePiece
        if self._using_sentencepiece:
            special_tokens = [
                self.pad_token,
                self.unk_token,
                self.bos_token,
                self.eos_token,
                self.mask_token
            ]
            for token in special_tokens:
                if token not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[token] = idx
                    self.ids_to_tokens[idx] = token
                    
        # Token IDs
        self.pad_token_id = self.vocab.get(self.pad_token, 0)
        self.unk_token_id = self.vocab.get(self.unk_token, 1)
        self.bos_token_id = self.vocab.get(self.bos_token, 2)
        self.eos_token_id = self.vocab.get(self.eos_token, 3)
        self.mask_token_id = self.vocab.get(self.mask_token, 4)
        
    def _load_vocab(self, vocab_file: Union[str, Path]) -> Dict[str, int]:
        """Load vocabulary from file."""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def train(
        self,
        files: List[str],
        vocab_size: int = 32000,
        model_prefix: str = "tokenizer",
        model_type: str = "bpe",
        character_coverage: float = 1.0,
        num_threads: int = 4
    ):
        """
        Train a new SentencePiece model.
        
        Args:
            files: List of text files for training
            vocab_size: Vocabulary size
            model_prefix: Prefix for saved model files
            model_type: Model type (bpe/unigram/char/word)
            character_coverage: Character coverage
            num_threads: Number of threads for training
        """
        if not SENTENCEPIECE_AVAILABLE:
            raise ImportError(
                "sentencepiece package is required for training. "
                "Install with: pip install sentencepiece"
            )
            
        # Train SentencePiece model
        spm.SentencePieceTrainer.Train(
            input=','.join(files),
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            num_threads=num_threads,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece=self.pad_token,
            unk_piece=self.unk_token,
            bos_piece=self.bos_token,
            eos_piece=self.eos_token,
            user_defined_symbols=[self.mask_token]
        )
        
        # Load the trained model
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(f"{model_prefix}.model")
        self._using_sentencepiece = True
        
        # Update vocabulary
        self.vocab = {
            self.sp.IdToPiece(i): i
            for i in range(self.sp.GetPieceSize())
        }
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        
    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: Optional[bool] = None,
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None
    ) -> Dict[str, Union[List[int], torch.Tensor, np.ndarray]]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text or list of texts
            add_special_tokens: Whether to add special tokens
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            return_tensors: Output tensor type ('pt', 'np' or None)
            
        Returns:
            Dictionary containing:
                input_ids: Token IDs
                attention_mask: Attention mask
        """
        if self._using_transformers:
            return self.tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length or self.max_length,
                return_tensors=return_tensors
            )
            
        # Process single string
        if isinstance(text, str):
            text = [text]
            
        # Encode each text
        encoded = []
        for t in text:
            if self._using_sentencepiece:
                ids = self.sp.EncodeAsIds(t)
            else:
                # Fallback to simple splitting
                tokens = t.split()
                ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
                
            if add_special_tokens or self.add_special_tokens:
                ids = [self.bos_token_id] + ids + [self.eos_token_id]
                
            encoded.append(ids)
            
        # Truncate if needed
        if truncation and max_length:
            encoded = [ids[:max_length] for ids in encoded]
            
        # Pad sequences
        if padding:
            max_len = max(len(ids) for ids in encoded)
            attention_mask = [
                [1] * len(ids) + [0] * (max_len - len(ids))
                for ids in encoded
            ]
            encoded = [
                ids + [self.pad_token_id] * (max_len - len(ids))
                for ids in encoded
            ]
            
        # Convert to tensors if requested
        if return_tensors == 'pt':
            encoded = torch.tensor(encoded)
            attention_mask = torch.tensor(attention_mask)
        elif return_tensors == 'np':
            encoded = np.array(encoded)
            attention_mask = np.array(attention_mask)
            
        return {
            'input_ids': encoded,
            'attention_mask': attention_mask
        }
        
    def decode(
        self,
        token_ids: Union[List[int], List[List[int]], torch.Tensor, np.ndarray],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ) -> Union[str, List[str]]:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to remove special tokens
            clean_up_tokenization_spaces: Whether to clean up spaces
            
        Returns:
            Decoded text or list of texts
        """
        if self._using_transformers:
            return self.tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces
            )
            
        # Convert tensor/array to list
        if isinstance(token_ids, (torch.Tensor, np.ndarray)):
            token_ids = token_ids.tolist()
            
        # Handle batch input
        if isinstance(token_ids[0], list):
            return [self.decode(ids) for ids in token_ids]
            
        if self._using_sentencepiece:
            # Use SentencePiece decoding
            if skip_special_tokens:
                token_ids = [
                    id for id in token_ids
                    if id not in [
                        self.pad_token_id,
                        self.bos_token_id,
                        self.eos_token_id,
                        self.mask_token_id
                    ]
                ]
            return self.sp.DecodeIds(token_ids)
            
        # Fallback to vocabulary decoding
        tokens = [self.ids_to_tokens.get(id, self.unk_token) for id in token_ids]
        
        if skip_special_tokens:
            tokens = [
                token for token in tokens
                if token not in [
                    self.pad_token,
                    self.bos_token,
                    self.eos_token,
                    self.mask_token
                ]
            ]
            
        text = ' '.join(tokens)
        
        if clean_up_tokenization_spaces:
            text = text.replace(' .', '.').replace(' ,', ',')
            text = text.replace(' ?', '?').replace(' !', '!')
            text = text.replace(' \'', '\'').replace(' n\'', 'n\'')
            text = text.replace('`` ', '"').replace(' \'\'', '"')
            
        return text
        
    def save_pretrained(self, save_directory: Union[str, Path]):
        """Save tokenizer files to directory."""
        if self._using_transformers:
            self.tokenizer.save_pretrained(save_directory)
            return
            
        os.makedirs(save_directory, exist_ok=True)
        
        if self._using_sentencepiece:
            # Save SentencePiece model
            self.sp.Save(os.path.join(save_directory, "tokenizer.model"))
        
        # Save vocabulary
        vocab_file = os.path.join(save_directory, 'vocab.json')
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f)
            
    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: Union[str, Path],
        **kwargs
    ) -> 'Tokenizer':
        """Load pretrained tokenizer."""
        if os.path.isfile(os.path.join(pretrained_path, "tokenizer.model")):
            return cls(
                spm_file=os.path.join(pretrained_path, "tokenizer.model"),
                **kwargs
            )
        elif os.path.isfile(os.path.join(pretrained_path, "vocab.json")):
            return cls(
                vocab_file=os.path.join(pretrained_path, "vocab.json"),
                **kwargs
            )
        else:
            return cls(pretrained_model_name=pretrained_path, **kwargs)
            
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        if self._using_transformers:
            return len(self.tokenizer)
        return len(self.vocab)
        
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary mapping."""
        if self._using_transformers:
            return self.tokenizer.get_vocab()
        return self.vocab.copy()
