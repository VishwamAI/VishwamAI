"""
Tokenizer implementation for Vishwamai model
"""
import os
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import json
import torch
from transformers import PreTrainedTokenizer
from sentencepiece import SentencePieceProcessor

class VishwamaiTokenizer(PreTrainedTokenizer):
    """
    Tokenizer for Vishwamai model using SentencePiece
    """
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        model_file: Optional[str] = None,
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        tree_tokens: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs
        )
        
        # Load SentencePiece model
        if model_file and os.path.exists(model_file):
            self.sp_model = SentencePieceProcessor()
            self.sp_model.Load(model_file)
        else:
            self.sp_model = None
            
        # Special tokens for tree planning
        self.tree_tokens = tree_tokens or [
            "<node>",
            "</node>",
            "<leaf>",
            "</leaf>",
            "<action>",
            "</action>"
        ]
        
        # Initialize vocabulary
        self.vocab_file = vocab_file
        self.merges_file = merges_file
        self.encoder = {}
        self.decoder = {}
        
        if vocab_file:
            self.load_vocab(vocab_file)
            
    def load_vocab(self, vocab_file: str) -> None:
        """Load vocabulary from file"""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
            
        self.encoder = {word: idx for idx, word in enumerate(vocab)}
        self.decoder = {idx: word for word, idx in self.encoder.items()}
        
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary mapping"""
        return self.encoder
        
    def save_vocabulary(
        self,
        save_directory: Union[str, Path],
        filename_prefix: Optional[str] = None
    ) -> Tuple[str, str]:
        """Save vocabulary files"""
        save_dir = Path(save_directory)
        files = []
        
        if self.vocab_file:
            vocab_file = save_dir / (
                f"{filename_prefix}-vocab.json" if filename_prefix else "vocab.json"
            )
            with open(vocab_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.encoder.keys()), f, ensure_ascii=False)
            files.append(str(vocab_file))
            
        if self.merges_file:
            merges_file = save_dir / (
                f"{filename_prefix}-merges.txt" if filename_prefix else "merges.txt"
            )
            # Save merges if using BPE
            files.append(str(merges_file))
            
        return tuple(files)
        
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using SentencePiece or fallback tokenization
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        if self.sp_model:
            return self.sp_model.EncodeAsPieces(text)
        
        # Fallback basic tokenization
        words = text.split()
        tokens = []
        for word in words:
            if word in self.encoder:
                tokens.append(word)
            else:
                tokens.append(self.unk_token)
        return tokens
        
    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to vocabulary id"""
        return self.encoder.get(token, self.encoder.get(self.unk_token))
        
    def _convert_id_to_token(self, index: int) -> str:
        """Convert vocabulary id to token"""
        return self.decoder.get(index, self.unk_token)
        
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert tokens back to string"""
        if self.sp_model:
            return self.sp_model.DecodePieces(tokens)
        return " ".join(tokens)
        
    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Build model inputs from token ids with special tokens"""
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]
        
    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Create token type IDs for sequence pairs"""
        if token_ids_1 is None:
            return [0] * len(token_ids_0)
            
        return [0] * len(token_ids_0) + [1] * len(token_ids_1)
        
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.encoder)
