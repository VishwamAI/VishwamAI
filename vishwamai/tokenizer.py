import os
from typing import List, Optional, Union
import sentencepiece as spm
import torch
from dataclasses import dataclass

@dataclass
class TokenizerConfig:
    """Configuration for VishwamAI tokenizer."""
    vocab_size: int = 102400
    model_prefix: str = "vishwamai"
    character_coverage: float = 0.9999
    num_threads: int = 8
    max_sentence_length: int = 16384
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3
    pad_token: str = "<|pad|>"
    bos_token: str = "<|startoftext|>"
    eos_token: str = "<|endoftext|>"
    unk_token: str = "<|unk|>"
    mask_token: str = "<|mask|>"
    special_tokens: List[str] = None

class VishwamAITokenizer:
    """
    SentencePiece tokenizer for VishwamAI.
    Handles tokenization with custom vocabulary and special tokens.
    """
    
    def __init__(self, config: TokenizerConfig = None):
        self.config = config or TokenizerConfig()
        self.sp_model = None
        self.special_tokens = {
            "pad_token": self.config.pad_token,
            "bos_token": self.config.bos_token,
            "eos_token": self.config.eos_token,
            "unk_token": self.config.unk_token,
            "mask_token": self.config.mask_token
        }
        if self.config.special_tokens:
            self.special_tokens.update({f"special_{i}": t for i, t in enumerate(self.config.special_tokens)})
    
    def train(self, texts: Union[str, List[str]], output_dir: str = "tokenizer"):
        """
        Train the tokenizer on input texts.
        
        Args:
            texts: Training texts or path to text file
            output_dir: Directory to save the tokenizer model
        """
        os.makedirs(output_dir, exist_ok=True)
        model_prefix = os.path.join(output_dir, self.config.model_prefix)
        
        # Prepare training data
        if isinstance(texts, list):
            train_path = os.path.join(output_dir, "train.txt")
            with open(train_path, "w", encoding="utf-8") as f:
                for text in texts:
                    f.write(text + "\n")
        else:
            train_path = texts
            
        # Train SentencePiece model
        spm.SentencePieceTrainer.train(
            input=train_path,
            model_prefix=model_prefix,
            vocab_size=self.config.vocab_size,
            character_coverage=self.config.character_coverage,
            num_threads=self.config.num_threads,
            pad_id=self.config.pad_id,
            bos_id=self.config.bos_id,
            eos_id=self.config.eos_id,
            unk_id=self.config.unk_id,
            max_sentence_length=self.config.max_sentence_length,
            user_defined_symbols=list(self.special_tokens.values())
        )
        
        # Load the trained model
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(f"{model_prefix}.model")
    
    def load(self, model_path: str):
        """Load a trained tokenizer model."""
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_path)
    
    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: Optional[str] = None
    ) -> Union[List[int], torch.Tensor]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text or list of texts
            add_special_tokens: Whether to add BOS/EOS tokens
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            return_tensors: Output format ("pt" for PyTorch tensors)
            
        Returns:
            Token IDs as list or tensor
        """
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
            
        encoded = []
        for t in texts:
            ids = self.sp_model.encode(t)
            if add_special_tokens:
                ids = [self.sp_model.bos_id()] + ids + [self.sp_model.eos_id()]
            encoded.append(ids)
            
        # Handle length constraints
        if max_length:
            if truncation:
                encoded = [ids[:max_length] for ids in encoded]
            if padding:
                encoded = [ids + [self.sp_model.pad_id()] * (max_length - len(ids)) 
                          for ids in encoded]
                
        if return_tensors == "pt":
            return torch.tensor(encoded)
        return encoded if len(encoded) > 1 else encoded[0]
    
    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True
    ) -> Union[str, List[str]]:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Input token IDs
            skip_special_tokens: Whether to remove special tokens
            
        Returns:
            Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        if isinstance(token_ids[0], list):
            return [self.sp_model.decode(ids) for ids in token_ids]
        return self.sp_model.decode(token_ids)
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.sp_model.get_piece_size()
    
    def get_special_tokens_mask(
        self,
        token_ids: Union[List[int], torch.Tensor],
        already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Get mask identifying special tokens.
        
        Args:
            token_ids: Input token IDs
            already_has_special_tokens: Whether input already has special tokens
            
        Returns:
            List of 1s for special tokens and 0s for normal tokens
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        mask = []
        for id in token_ids:
            if (id == self.sp_model.pad_id() or
                id == self.sp_model.bos_id() or
                id == self.sp_model.eos_id() or
                id == self.sp_model.unk_id()):
                mask.append(1)
            else:
                mask.append(0)
        return mask
