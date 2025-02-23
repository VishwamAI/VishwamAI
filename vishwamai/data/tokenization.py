"""Tokenization module using SentencePiece for VishwamAI."""

import os
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import yaml
import logging
import tempfile
import sentencepiece as spm
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

class SentencePieceTokenizer:
    """Handles tokenization using SentencePiece model."""
    
    def __init__(self, config_path: Union[str, Path]):
        """Initialize tokenizer with configuration.
        
        Args:
            config_path: Path to data_config.yaml
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)["tokenizer"]
            
        self.vocab_size = self.config["vocab_size"]
        self.model_type = self.config["model_type"]
        self.character_coverage = self.config["character_coverage"]
        self.padding_side = self.config["padding_side"]
        self.truncation_side = self.config["truncation_side"]
        
        self.sp_model = None
        self.special_tokens = self.config["special_tokens"]
        
    def train(self, texts: List[str], output_dir: Union[str, Path],
              model_prefix: str = "tokenizer") -> None:
        """Train SentencePiece tokenizer on input texts.
        
        Args:
            texts: List of training texts
            output_dir: Directory to save tokenizer model
            model_prefix: Prefix for tokenizer files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write training data to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            for text in texts:
                f.write(text + '\n')
            train_file = f.name
            
        try:
            # Train tokenizer
            model_path = str(output_dir / model_prefix)
            
            spm.SentencePieceTrainer.train(
                input=train_file,
                model_prefix=model_path,
                vocab_size=self.vocab_size,
                model_type=self.model_type,
                character_coverage=self.character_coverage,
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
                pad_piece=self.special_tokens["pad_token"],
                unk_piece=self.special_tokens["unk_token"],
                bos_piece=self.special_tokens["bos_token"],
                eos_piece=self.special_tokens["eos_token"],
                user_defined_symbols=[self.special_tokens["mask_token"]]
            )
            
            # Load trained model
            self.load(model_path + ".model")
            
        finally:
            # Clean up temporary file
            os.unlink(train_file)
            
    def load(self, model_path: Union[str, Path]) -> None:
        """Load pretrained SentencePiece model.
        
        Args:
            model_path: Path to .model file
        """
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(str(model_path))
        
    def save(self, output_dir: Union[str, Path],
             model_prefix: str = "tokenizer") -> None:
        """Save tokenizer model and vocabulary.
        
        Args:
            output_dir: Directory to save files
            model_prefix: Prefix for saved files
        """
        if self.sp_model is None:
            raise ValueError("No model to save. Train or load a model first.")
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = str(output_dir / f"{model_prefix}.model")
        self.sp_model.save(model_path)
        
        # Save vocab
        vocab_path = output_dir / f"{model_prefix}.vocab"
        with open(vocab_path, 'w') as f:
            for i in range(self.sp_model.get_piece_size()):
                piece = self.sp_model.id_to_piece(i)
                score = self.sp_model.get_score(i)
                f.write(f"{piece}\t{score}\n")
                
    def encode(self, text: str, 
              add_special_tokens: bool = True,
              max_length: Optional[int] = None) -> List[int]:
        """Encode text to token ids.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
            max_length: Maximum sequence length
            
        Returns:
            List of token ids
        """
        if self.sp_model is None:
            raise ValueError("No model loaded. Train or load a model first.")
            
        if not text:
            return []
            
        # Tokenize
        token_ids = self.sp_model.encode(text)
        
        # Add special tokens if requested
        if add_special_tokens:
            bos_id = self.sp_model.piece_to_id(self.special_tokens["bos_token"])
            eos_id = self.sp_model.piece_to_id(self.special_tokens["eos_token"])
            token_ids = [bos_id] + token_ids + [eos_id]
            
        # Truncate if needed
        if max_length is not None:
            if self.truncation_side == "right":
                token_ids = token_ids[:max_length]
            else:
                token_ids = token_ids[-max_length:]
                
        return token_ids
        
    def decode(self, token_ids: List[int],
              skip_special_tokens: bool = True) -> str:
        """Decode token ids back to text.
        
        Args:
            token_ids: List of token ids
            skip_special_tokens: Whether to remove special tokens
            
        Returns:
            Decoded text
        """
        if self.sp_model is None:
            raise ValueError("No model loaded. Train or load a model first.")
            
        if not token_ids:
            return ""
            
        # Convert to list if numpy array
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
            
        # Filter special tokens if requested
        if skip_special_tokens:
            special_ids = [self.sp_model.piece_to_id(token)
                         for token in self.special_tokens.values()]
            token_ids = [id for id in token_ids if id not in special_ids]
            
        return self.sp_model.decode(token_ids)
        
    def encode_batch(self, texts: List[str],
                    add_special_tokens: bool = True,
                    max_length: Optional[int] = None,
                    padding: bool = True) -> Tuple[List[List[int]], List[List[int]]]:
        """Encode a batch of texts.
        
        Args:
            texts: List of input texts
            add_special_tokens: Whether to add BOS/EOS tokens
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            
        Returns:
            Tuple of (token_ids, attention_mask)
        """
        # Encode all texts
        batch_tokens = [
            self.encode(text, add_special_tokens=add_special_tokens,
                       max_length=max_length)
            for text in tqdm(texts, desc="Encoding texts")
        ]
        
        # Find max length in batch if no max_length specified
        if max_length is None:
            max_length = max(len(tokens) for tokens in batch_tokens)
            
        # Initialize attention masks
        attention_masks = []
        
        # Pad sequences if requested
        if padding:
            pad_id = self.sp_model.piece_to_id(self.special_tokens["pad_token"])
            
            padded_tokens = []
            for tokens in batch_tokens:
                padding_length = max_length - len(tokens)
                
                if self.padding_side == "right":
                    padded = tokens + [pad_id] * padding_length
                    mask = [1] * len(tokens) + [0] * padding_length
                else:
                    padded = [pad_id] * padding_length + tokens
                    mask = [0] * padding_length + [1] * len(tokens)
                    
                padded_tokens.append(padded)
                attention_masks.append(mask)
                
            batch_tokens = padded_tokens
        else:
            # If no padding, attention mask is all ones
            attention_masks = [[1] * len(tokens) for tokens in batch_tokens]
            
        return batch_tokens, attention_masks
        
    def get_vocab_size(self) -> int:
        """Get vocabulary size.
        
        Returns:
            Size of vocabulary
        """
        if self.sp_model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        return self.sp_model.get_piece_size()
        
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary mapping.
        
        Returns:
            Dictionary mapping tokens to ids
        """
        if self.sp_model is None:
            raise ValueError("No model loaded. Train or load a model first.")
            
        vocab = {}
        for i in range(self.sp_model.get_piece_size()):
            piece = self.sp_model.id_to_piece(i)
            vocab[piece] = i
        return vocab
        
    def token_to_id(self, token: str) -> int:
        """Convert token to id.
        
        Args:
            token: Input token
            
        Returns:
            Token id
        """
        if self.sp_model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        return self.sp_model.piece_to_id(token)
        
    def id_to_token(self, id: int) -> str:
        """Convert id to token.
        
        Args:
            id: Token id
            
        Returns:
            Token string
        """
        if self.sp_model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        return self.sp_model.id_to_piece(id)
