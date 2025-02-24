import os
import json
import logging
import shutil
from typing import List, Optional, Union, Dict
from pathlib import Path
import sentencepiece as spm

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
        character_coverage: float = 0.99995,
        model_type: str = "bpe",
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
            character_coverage (float): The character coverage for training.
            model_type (str): The type of model (e.g., "bpe").
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

    @property
    def vocab_size(self) -> int:
        """
        Get the vocabulary size.

        Returns:
            int: The vocabulary size.
        """
        return self.sp_model.get_piece_size() if self.sp_model else self._vocab_size

    def train(self, input_files: Union[str, List[str]], output_dir: str) -> None:
        """
        Train the SentencePiece tokenizer on input files.

        Args:
            input_files (Union[str, List[str]]): The input files for training.
            output_dir (str): The directory to save the trained model.

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
            "user_defined_symbols": ["<mask>", "<sep>", "<cls>"],
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
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def encode(self, text: Union[str, List[str]], add_special_tokens: bool = True) -> Union[List[int], List[List[int]]]:
        """
        Encode text to token ids.

        Args:
            text (Union[str, List[str]]): The text to encode.
            add_special_tokens (bool): Whether to add special tokens.

        Returns:
            Union[List[int], List[List[int]]]: The encoded token ids.

        Raises:
            RuntimeError: If model is not loaded or encoding fails.
        """
        if self.sp_model is None:
            raise RuntimeError("Tokenizer model not loaded. Call load() or train() first.")
            
        if isinstance(text, str):
            text = [text]
        
        try:
            encoded = []
            for t in text:
                if add_special_tokens:
                    ids = [self.bos_id] + self.sp_model.encode_as_ids(t) + [self.eos_id]
                else:
                    ids = self.sp_model.encode_as_ids(t)
                encoded.append(ids)
                
            return encoded[0] if len(encoded) == 1 else encoded
        except Exception as e:
            raise RuntimeError(f"Encoding failed: {str(e)}")

    def decode(self, token_ids: Union[List[int], List[List[int]]], skip_special_tokens: bool = True) -> Union[str, List[str]]:
        """
        Decode token ids to text.

        Args:
            token_ids (Union[List[int], List[List[int]]]): The token ids to decode.
            skip_special_tokens (bool): Whether to skip special tokens.

        Returns:
            Union[str, List[str]]: The decoded text.

        Raises:
            RuntimeError: If model is not loaded or decoding fails.
        """
        if self.sp_model is None:
            raise RuntimeError("Tokenizer model not loaded. Call load() or train() first.")
            
        if isinstance(token_ids[0], int):
            token_ids = [token_ids]
            
        try:
            decoded = []
            for ids in token_ids:
                if skip_special_tokens:
                    ids = [id for id in ids if id not in {self.pad_id, self.eos_id, self.bos_id}]
                text = self.sp_model.decode_ids(ids)
                decoded.append(text)
                
            return decoded[0] if len(decoded) == 1 else decoded
        except Exception as e:
            raise RuntimeError(f"Decoding failed: {str(e)}")

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
            
            # Save the configuration
            config = {
                "vocab_size": self._vocab_size,
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
                
            tokenizer = cls(**config)
            model_path = os.path.join(model_dir, f"{config['model_prefix']}.model")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            tokenizer.load(model_path)
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
            
        return {
            self.sp_model.id_to_piece(id): id
            for id in range(self.sp_model.get_piece_size())
        }
