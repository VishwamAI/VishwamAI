import os
import json
from typing import List, Optional, Union, Dict
from pathlib import Path
import sentencepiece as spm
import numpy as np

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
        self.vocab_size = vocab_size 
        self.model_prefix = model_prefix
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.unk_id = unk_id
        self.bos_id = bos_id
        self.character_coverage = character_coverage
        self.model_type = model_type
        self.sp_model = None

    def train(self, input_files: Union[str, List[str]], output_dir: str) -> None:
        """Train the SentencePiece tokenizer on input files"""
        if isinstance(input_files, str):
            input_files = [input_files]
            
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Prepare training arguments
        train_args = {
            "input": ",".join(input_files),
            "model_prefix": os.path.join(output_dir, self.model_prefix),
            "vocab_size": self.vocab_size,
            "character_coverage": self.character_coverage,
            "pad_id": self.pad_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
            "bos_id": self.bos_id,
            "model_type": self.model_type,
            "user_defined_symbols": ["<mask>", "<sep>", "<cls>"],
            "control_symbols": ["<start>", "<end>"],
            "normalization_rule_name": "nmt_nfkc_cf"
        }
        
        # Train the model
        spm.SentencePieceTrainer.train(**train_args)
        
        # Load the trained model
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(f"{train_args['model_prefix']}.model")

    def load(self, model_path: str) -> None:
        """Load a pretrained SentencePiece model"""
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_path)

    def encode(self, text: Union[str, List[str]], add_special_tokens: bool = True) -> Union[List[int], List[List[int]]]:
        """Encode text to token ids"""
        if isinstance(text, str):
            text = [text]
        
        encoded = []
        for t in text:
            if add_special_tokens:
                ids = [self.bos_id] + self.sp_model.encode_as_ids(t) + [self.eos_id]
            else:
                ids = self.sp_model.encode_as_ids(t)
            encoded.append(ids)
            
        return encoded[0] if len(encoded) == 1 else encoded

    def decode(self, token_ids: Union[List[int], List[List[int]]], skip_special_tokens: bool = True) -> Union[str, List[str]]:
        """Decode token ids to text"""
        if isinstance(token_ids[0], int):
            token_ids = [token_ids]
            
        decoded = []
        for ids in token_ids:
            if skip_special_tokens:
                ids = [id for id in ids if id not in {self.pad_id, self.eos_id, self.bos_id}]
            text = self.sp_model.decode_ids(ids)
            decoded.append(text)
            
        return decoded[0] if len(decoded) == 1 else decoded

    def save(self, output_dir: str) -> None:
        """Save the tokenizer files"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save the model file
        if self.sp_model is not None:
            model_path = os.path.join(output_dir, f"{self.model_prefix}.model")
            vocab_path = os.path.join(output_dir, f"{self.model_prefix}.vocab")
            self.sp_model.save_model(model_path)
            
        # Save the configuration
        config = {
            "vocab_size": self.vocab_size,
            "model_prefix": self.model_prefix,
            "pad_id": self.pad_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
            "bos_id": self.bos_id,
            "character_coverage": self.character_coverage,
            "model_type": self.model_type
        }
        
        config_path = os.path.join(output_dir, "tokenizer_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "VishwamAITokenizer":
        """Load a pretrained tokenizer from directory"""
        config_path = os.path.join(model_dir, "tokenizer_config.json")
        with open(config_path) as f:
            config = json.load(f)
            
        tokenizer = cls(**config)
        model_path = os.path.join(model_dir, f"{config['model_prefix']}.model")
        tokenizer.load(model_path)
        return tokenizer

    def get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary mapping"""
        return {
            self.sp_model.id_to_piece(id): id
            for id in range(self.sp_model.get_piece_size())
        }

    def vocab_size(self) -> int:
        """Get the vocabulary size"""
        return self.sp_model.get_piece_size() if self.sp_model else self.vocab_size
