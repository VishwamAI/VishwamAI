import sentencepiece as spm
import numpy as np
from typing import List, Union, Dict
from functools import lru_cache
import os
from huggingface_hub import snapshot_download

class VishwamAITokenizer:
    def __init__(
        self, 
        model_path: str,
        max_length: int = 512,
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        bos_token: str = "[BOS]",
        eos_token: str = "[EOS]"
    ):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.max_length = max_length
        
        # Special tokens
        self.special_tokens = {
            "pad_token": pad_token,
            "unk_token": unk_token,
            "bos_token": bos_token,
            "eos_token": eos_token
        }
        self._init_special_tokens()
        
        # Cache for frequently used tokens
        self.token_cache = {}
    
    def _init_special_tokens(self):
        """Initialize special token IDs"""
        for token in self.special_tokens.values():
            if not self.sp.piece_to_id(token):
                print(f"Warning: {token} not found in vocabulary")
    
    @lru_cache(maxsize=10000)
    def encode(
        self, 
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        padding: bool = True,
        truncation: bool = True
    ) -> np.ndarray:
        """Encode text with caching and advanced options"""
        if isinstance(text, str):
            text = [text]
            
        encoded = []
        for t in text:
            tokens = self.sp.encode_as_ids(t)
            if add_special_tokens:
                tokens = [self.sp.piece_to_id(self.special_tokens["bos_token"])] + \
                        tokens + \
                        [self.sp.piece_to_id(self.special_tokens["eos_token"])]
            
            if truncation:
                tokens = tokens[:self.max_length]
                
            if padding:
                pad_length = self.max_length - len(tokens)
                if pad_length > 0:
                    tokens.extend([self.sp.piece_to_id(self.special_tokens["pad_token"])] * pad_length)
                    
            encoded.append(tokens)
            
        return np.array(encoded)
    
    def decode(self, ids: Union[List[int], np.ndarray], skip_special_tokens: bool = True) -> str:
        """Decode with special token handling"""
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
            
        if skip_special_tokens:
            special_ids = [self.sp.piece_to_id(token) for token in self.special_tokens.values()]
            ids = [id for id in ids if id not in special_ids]
            
        return self.sp.decode_ids(ids)
    
    def batch_encode(self, texts: List[str], batch_size: int = 32, **kwargs) -> List[np.ndarray]:
        """Process texts in batches to manage memory"""
        batches = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded_batch = self.encode(batch, **kwargs)
            batches.append(encoded_batch)
        return np.concatenate(batches, axis=0)
    
    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()
    
    def save_pretrained(self, path: str):
        """Save tokenizer configuration"""
        self.sp.save(f"{path}/tokenizer.model")
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """Load tokenizer from HuggingFace Hub or local directory"""
        # Download tokenizer if not local
        if not os.path.isdir(model_name_or_path):
            model_path = snapshot_download(
                repo_id=model_name_or_path,
                allow_patterns=["tokenizer.model", "tokenizer.json"],
                local_files_only=False
            )
        else:
            model_path = model_name_or_path
            
        # Find tokenizer model file
        tokenizer_path = os.path.join(model_path, "tokenizer.model")
        if not os.path.exists(tokenizer_path):
            raise ValueError(f"No tokenizer.model found in {model_path}")
            
        return cls(
            model_path=tokenizer_path,
            **kwargs
        )
