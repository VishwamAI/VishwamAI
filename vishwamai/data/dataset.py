"""
Dataset implementation for Vishwamai model
"""
import os
from typing import Dict, List, Optional, Union, Callable
import torch
from torch.utils.data import Dataset
import json
import logging

from .tokenizer import VishwamaiTokenizer

logger = logging.getLogger(__name__)

class VishwamaiDataset(Dataset):
    """
    Dataset for training Vishwamai model
    """
    def __init__(
        self,
        data_path: Union[str, List[str]],
        tokenizer: VishwamaiTokenizer,
        max_length: int = 1024,
        preprocessing_fn: Optional[Callable] = None,
        cache_dir: Optional[str] = None,
        is_training: bool = True
    ):
        self.data_path = data_path if isinstance(data_path, list) else [data_path]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessing_fn = preprocessing_fn
        self.cache_dir = cache_dir
        self.is_training = is_training
        
        # Load and process data
        self.examples = []
        self._load_data()
        
    def _load_data(self) -> None:
        """Load and preprocess data from files"""
        # Check cache first
        if self.cache_dir and os.path.exists(self.cache_dir):
            cache_file = os.path.join(
                self.cache_dir,
                f"processed_data_{self.max_length}.pt"
            )
            if os.path.exists(cache_file):
                logger.info(f"Loading cached data from {cache_file}")
                self.examples = torch.load(cache_file)
                return
        
        # Load from data files
        for data_file in self.data_path:
            logger.info(f"Loading data from {data_file}")
            
            if data_file.endswith('.json'):
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif data_file.endswith('.jsonl'):
                data = []
                with open(data_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
            else:
                raise ValueError(f"Unsupported file format: {data_file}")
            
            # Process examples
            for item in data:
                processed = self._process_example(item)
                if processed is not None:
                    self.examples.append(processed)
                    
        # Cache processed data
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_file = os.path.join(
                self.cache_dir,
                f"processed_data_{self.max_length}.pt"
            )
            torch.save(self.examples, cache_file)
            logger.info(f"Cached processed data to {cache_file}")
            
    def _process_example(
        self,
        example: Dict
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Process single data example
        
        Args:
            example: Raw data example
            
        Returns:
            Processed example with tensors or None if invalid
        """
        # Apply custom preprocessing if provided
        if self.preprocessing_fn is not None:
            example = self.preprocessing_fn(example)
            
        # Extract text fields
        input_text = example.get('input', '')
        target_text = example.get('target', '')
        
        if not input_text or (self.is_training and not target_text):
            return None
            
        # Tokenize
        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Add labels for training
        if self.is_training and target_text:
            labels = self.tokenizer(
                target_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            model_inputs['labels'] = labels['input_ids']
            
        # Convert to tensors and squeeze batch dimension
        processed = {
            k: v.squeeze(0) for k, v in model_inputs.items()
        }
        
        # Add metadata if present
        for key in ['id', 'type', 'metadata']:
            if key in example:
                processed[key] = example[key]
                
        return processed
        
    def __len__(self) -> int:
        """Get dataset size"""
        return len(self.examples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get example by index"""
        return self.examples[idx]
        
    def filter_by_length(self, max_length: int) -> None:
        """Filter examples by sequence length"""
        original_size = len(self.examples)
        self.examples = [
            ex for ex in self.examples
            if len(ex['input_ids']) <= max_length
        ]
        logger.info(
            f"Filtered dataset from {original_size} to {len(self.examples)} "
            f"examples with max length {max_length}"
        )
        
    def filter_by_type(self, allowed_types: List[str]) -> None:
        """Filter examples by type"""
        if not any('type' in ex for ex in self.examples):
            return
            
        original_size = len(self.examples)
        self.examples = [
            ex for ex in self.examples
            if ex.get('type', '') in allowed_types
        ]
        logger.info(
            f"Filtered dataset from {original_size} to {len(self.examples)} "
            f"examples with types {allowed_types}"
        )
