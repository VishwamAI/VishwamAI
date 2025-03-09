"""
Dataset loader and processing utilities for VishwamAI.
Optimized for TPU and GPU training with efficient batching.
"""

import json
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Union, Any
import numpy as np
from functools import partial
import multiprocessing as mp

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

logger = logging.getLogger(__name__)

class VishwamAIDataset(Dataset):
    """Device-optimized dataset for VishwamAI training"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        mode: str = "normal",
        max_length: int = 512,
        cache_tokenization: bool = True,
        device_type: Optional[str] = None
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization
        
        # Auto-detect device type if not specified
        if device_type is None:
            if HAS_JAX and jax.devices("tpu"):
                self.device_type = "tpu"
            elif torch.cuda.is_available():
                self.device_type = "gpu"
            else:
                self.device_type = "cpu"
        else:
            self.device_type = device_type
            
        # Load and preprocess data
        logger.info(f"Loading dataset from {data_path} in {mode} mode")
        with open(data_path) as f:
            self.data = json.load(f)
            
        # Prepare parallel processing pool
        self.pool = mp.Pool(mp.cpu_count()) if cache_tokenization else None
            
        # Cache tokenization if enabled
        if cache_tokenization:
            logger.info("Pre-tokenizing dataset...")
            self.cached_encodings = self._parallel_tokenize()
            logger.info("Tokenization completed")
            
    def _parallel_tokenize(self) -> List[Dict]:
        """Tokenize examples in parallel"""
        tokenize_fn = partial(
            self._tokenize_example,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            mode=self.mode
        )
        return list(self.pool.imap(tokenize_fn, self.data))
    
    @staticmethod
    def _tokenize_example(example: Dict, tokenizer: Any, max_length: int, mode: str) -> Dict:
        """Tokenize a single example based on training mode"""
        if mode == "normal":
            return {
                "input_ids": tokenizer.encode(
                    example["input"],
                    max_length=max_length,
                    truncation=True,
                    padding="max_length"
                ),
                "labels": tokenizer.encode(
                    example["output"],
                    max_length=max_length,
                    truncation=True,
                    padding="max_length"
                )
            }
        elif mode == "cot":
            # Chain of Thought format
            thought = example.get("thought", "")
            answer = example.get("answer", "")
            cot_text = (
                f"{example['input']} {tokenizer.special_token_ids['think']} "
                f"{thought} {tokenizer.special_token_ids['think_end']} "
                f"{tokenizer.special_token_ids['answer']} {answer} "
                f"{tokenizer.special_token_ids['answer_end']}"
            )
            encoded = tokenizer.encode(
                cot_text,
                max_length=max_length,
                truncation=True,
                padding="max_length"
            )
            return {
                "input_ids": tokenizer.encode(
                    example["input"],
                    max_length=max_length,
                    truncation=True,
                    padding="max_length"
                ),
                "labels": encoded,
                "thought_positions": VishwamAIDataset._get_special_token_positions(
                    encoded,
                    [
                        tokenizer.special_token_ids["think"],
                        tokenizer.special_token_ids["think_end"]
                    ]
                )
            }
        elif mode == "tot":
            # Tree of Thoughts format
            encoded = tokenizer.encode(
                example["input"],
                max_length=max_length,
                truncation=True,
                padding="max_length"
            )
            return {
                "input_ids": encoded,
                "labels": tokenizer.encode(
                    example["output"],
                    max_length=max_length,
                    truncation=True,
                    padding="max_length"
                ),
                "thought_tree": example.get("thought_tree", None),
                "target_tree": example.get("target_tree", None)
            }
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
    @staticmethod
    def _get_special_token_positions(token_ids: List[int], special_tokens: List[int]) -> List[int]:
        """Get positions of special tokens in sequence"""
        positions = []
        for i, token_id in enumerate(token_ids):
            if token_id in special_tokens:
                positions.append(i)
        return positions
            
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized example with device-specific optimizations"""
        if self.cache_tokenization:
            item = self.cached_encodings[idx]
        else:
            item = self._tokenize_example(
                self.data[idx],
                self.tokenizer,
                self.max_length,
                self.mode
            )
            
        # Convert to appropriate tensor type
        if self.device_type == "tpu" and HAS_JAX:
            return {k: jnp.array(v) for k, v in item.items() if not isinstance(v, (list, dict))}
        else:
            return {k: torch.tensor(v) for k, v in item.items() if not isinstance(v, (list, dict))}

def create_dataloader(
    dataset: VishwamAIDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Union[DataLoader, Any]:
    """Create appropriate dataloader based on device type"""
    
    if dataset.device_type == "tpu" and HAS_JAX:
        # Use JAX-specific data loading
        return _create_jax_dataloader(dataset, batch_size, shuffle)
    else:
        # Use PyTorch DataLoader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            collate_fn=_collate_fn
        )

def _create_jax_dataloader(dataset: VishwamAIDataset, batch_size: int, shuffle: bool) -> Any:
    """Create a JAX-compatible data loader"""
    # Implementation depends on specific TPU setup
    raise NotImplementedError("JAX dataloader not implemented yet")

def _collate_fn(examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching"""
    batch = {}
    
    # Handle tensor data
    for key in examples[0].keys():
        if isinstance(examples[0][key], (torch.Tensor, np.ndarray)):
            batch[key] = torch.stack([ex[key] for ex in examples])
        else:
            # Pass through non-tensor data (e.g., thought trees)
            batch[key] = [ex[key] for ex in examples]
            
    return batch