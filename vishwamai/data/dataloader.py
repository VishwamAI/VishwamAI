"""Data loading and batching module for VishwamAI."""

import os
from typing import List, Dict, Optional, Union, Tuple, Any
from pathlib import Path
import yaml
import logging
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import numpy as np
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torch
from tqdm import tqdm
import mmap
import hashlib

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages caching of preprocessed and tokenized data."""
    
    def __init__(self, config: Dict, cache_dir: Union[str, Path]):
        """Initialize cache manager.
        
        Args:
            config: Caching configuration
            cache_dir: Directory for cache files
        """
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_cache_path(self, data_id: str, cache_type: str) -> Path:
        """Get path for cache file.
        
        Args:
            data_id: Unique identifier for data
            cache_type: Type of cached data (preprocessing/tokenization)
            
        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{data_id}_{cache_type}.cache"
        
    def save_cache(self, data: Any, data_id: str,
                  cache_type: str) -> None:
        """Save data to cache.
        
        Args:
            data: Data to cache
            data_id: Unique identifier for data
            cache_type: Type of cached data
        """
        cache_path = self.get_cache_path(data_id, cache_type)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
            
    def load_cache(self, data_id: str, cache_type: str) -> Optional[Any]:
        """Load data from cache if exists.
        
        Args:
            data_id: Unique identifier for data
            cache_type: Type of cached data
            
        Returns:
            Cached data if exists, else None
        """
        cache_path = self.get_cache_path(data_id, cache_type)
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
        
    def cache_exists(self, data_id: str, cache_type: str) -> bool:
        """Check if cache exists.
        
        Args:
            data_id: Unique identifier for data
            cache_type: Type of cached data
            
        Returns:
            True if cache exists
        """
        return self.get_cache_path(data_id, cache_type).exists()

class BaseDataset(Dataset):
    """Base dataset class for all data sources."""
    
    def __init__(self, texts: List[str], labels: Optional[List[Any]] = None,
                 preprocessor: Any = None, tokenizer: Any = None,
                 cache_manager: Optional[CacheManager] = None):
        """Initialize dataset.
        
        Args:
            texts: List of input texts
            labels: Optional list of labels
            preprocessor: Text preprocessor instance
            tokenizer: Tokenizer instance
            cache_manager: Cache manager instance
        """
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.cache_manager = cache_manager
        
        # Generate unique ID for this dataset
        self.data_id = self._generate_data_id()
        
        # Process data
        self._process_data()
        
    def _generate_data_id(self) -> str:
        """Generate unique ID for dataset based on content."""
        content = ''.join(self.texts[:100])  # Use first 100 texts
        return hashlib.md5(content.encode()).hexdigest()
        
    def _process_data(self) -> None:
        """Process raw texts through preprocessing and tokenization."""
        # Try loading from cache first
        if self.cache_manager is not None:
            processed_data = self.cache_manager.load_cache(
                self.data_id, "processed")
            if processed_data is not None:
                self.processed_texts, self.token_ids, self.attention_masks = processed_data
                return
                
        # Preprocess texts
        if self.preprocessor is not None:
            self.processed_texts = self.preprocessor.preprocess_batch(self.texts)
        else:
            self.processed_texts = self.texts
            
        # Tokenize texts
        if self.tokenizer is not None:
            self.token_ids, self.attention_masks = self.tokenizer.encode_batch(
                self.processed_texts)
        else:
            self.token_ids = self.processed_texts
            self.attention_masks = [[1] * len(text) for text in self.texts]
            
        # Cache processed data
        if self.cache_manager is not None:
            self.cache_manager.save_cache(
                (self.processed_texts, self.token_ids, self.attention_masks),
                self.data_id, "processed")
                
    def __len__(self) -> int:
        return len(self.texts)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary containing input_ids, attention_mask, and labels
        """
        item = {
            'input_ids': torch.tensor(self.token_ids[idx]),
            'attention_mask': torch.tensor(self.attention_masks[idx])
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
            
        return item

class MLDataset(BaseDataset):
    """Dataset for machine learning tasks (MMLU, MMMU, GSM8K)."""
    
    def __init__(self, data_path: Union[str, Path], split: str,
                 config: Dict, preprocessor: Any = None,
                 tokenizer: Any = None, cache_manager: Optional[CacheManager] = None):
        """Initialize ML dataset.
        
        Args:
            data_path: Path to data directory
            split: Data split (train/val/test)
            config: Dataset configuration
            preprocessor: Text preprocessor instance
            tokenizer: Tokenizer instance
            cache_manager: Cache manager instance
        """
        self.data_path = Path(data_path)
        self.split = split
        self.config = config
        
        # Load raw data
        texts, labels = self._load_data()
        
        # Initialize base dataset
        super().__init__(texts, labels, preprocessor, tokenizer, cache_manager)
        
    def _load_data(self) -> Tuple[List[str], List[Any]]:
        """Load data from files.
        
        Returns:
            Tuple of (texts, labels)
        """
        raise NotImplementedError("Subclasses must implement _load_data")

class MMLUDataset(MLDataset):
    """Dataset for MMLU (Massive Multitask Language Understanding)."""
    
    def _load_data(self) -> Tuple[List[str], List[Any]]:
        """Load MMLU data.
        
        Returns:
            Tuple of (texts, labels)
        """
        split_path = self.data_path / f"{self.split}.jsonl"
        
        texts, labels = [], []
        with open(split_path) as f:
            for line in f:
                item = json.loads(line)
                text = item['text']
                if self.config["include_explanations"] and 'explanation' in item:
                    text += f"\nExplanation: {item['explanation']}"
                texts.append(text)
                labels.append(item['label'])
                
        return texts, labels

class MMMUDataset(MLDataset):
    """Dataset for MMMU (Massive Multitask Mathematical Understanding)."""
    
    def _load_data(self) -> Tuple[List[str], List[Any]]:
        """Load MMMU data.
        
        Returns:
            Tuple of (texts, labels)
        """
        split_path = self.data_path / f"{self.split}.jsonl"
        
        texts, labels = [], []
        with open(split_path) as f:
            for line in f:
                item = json.loads(line)
                if item['difficulty'] in self.config["difficulty_levels"]:
                    text = item['question']
                    if self.config["include_solutions"] and 'solution' in item:
                        text += f"\nSolution: {item['solution']}"
                    texts.append(text)
                    labels.append(item['answer'])
                    
        return texts, labels

class GSM8KDataset(MLDataset):
    """Dataset for GSM8K (Grade School Math 8K)."""
    
    def _load_data(self) -> Tuple[List[str], List[Any]]:
        """Load GSM8K data.
        
        Returns:
            Tuple of (texts, labels)
        """
        split_path = self.data_path / f"{self.split}.jsonl"
        
        texts, labels = [], []
        with open(split_path) as f:
            for line in f:
                item = json.loads(line)
                text = item['question']
                if self.config["include_chain_of_thought"] and 'steps' in item:
                    text += "\nLet's solve this step by step:\n"
                    text += '\n'.join(item['steps'])
                texts.append(text[:self.config["max_sequence_length"]])
                labels.append(float(item['answer']))
                
        return texts, labels

def create_dataloader(dataset: Dataset, config: Dict) -> DataLoader:
    """Create DataLoader with proper configuration.
    
    Args:
        dataset: Dataset instance
        config: DataLoader configuration
        
    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        drop_last=config["drop_last"],
        shuffle=config["shuffle_train"],
        prefetch_factor=config["prefetch_factor"],
        persistent_workers=config["persistent_workers"]
    )
