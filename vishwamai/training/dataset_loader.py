"""
Dataset loading utilities with optional smallpond support for distributed processing.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any, Union
import pandas as pd
import numpy as np
from pathlib import Path

from vishwamai.training.smallpond_manager import SmallpondManager

class VishwamAIDataset(Dataset):
    """Dataset class with optional smallpond support"""
    
    def __init__(self,
                data_path: Union[str, Path],
                tokenizer,
                mode: str = 'normal',
                smallpond_config: Optional[Dict[str, Any]] = None,
                cache_dir: Optional[str] = None):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to data file/directory
            tokenizer: Tokenizer for text processing
            mode: Dataset mode ('normal', 'cot', or 'tot')
            smallpond_config: Optional smallpond configuration
            cache_dir: Directory for caching
        """
        self.data_path = str(Path(data_path).absolute())
        self.tokenizer = tokenizer
        self.mode = mode
        self.smallpond_manager = None
        
        if smallpond_config:
            self.smallpond_manager = SmallpondManager(
                config=smallpond_config,
                cache_dir=cache_dir
            )
            self.data = self.smallpond_manager.load_dataset(self.data_path)
        else:
            # Load data using pandas
            if self.data_path.endswith('.parquet'):
                self.data = pd.read_parquet(self.data_path)
            elif self.data_path.endswith('.csv'):
                self.data = pd.read_csv(self.data_path)
            else:
                raise ValueError(f"Unsupported file format: {self.data_path}")
                
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        """Get dataset item with appropriate processing based on mode"""
        row = self.data.iloc[idx]
        
        if self.mode == 'normal':
            return self._process_normal(row)
        elif self.mode == 'cot':
            return self._process_cot(row)
        elif self.mode == 'tot':
            return self._process_tot(row)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
    def _process_normal(self, row):
        """Process row for normal training"""
        input_ids = self.tokenizer.encode(
            row['input_text'],
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )
        
        labels = self.tokenizer.encode(
            row['target_text'],
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )
        
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
        
        return {
            'input_ids': input_ids.squeeze(0),
            'attention_mask': attention_mask.squeeze(0),
            'labels': labels.squeeze(0)
        }
        
    def _process_cot(self, row):
        """Process row for chain-of-thought training"""
        # Add special tokens for thought process
        input_text = row['input_text']
        thought_process = row.get('thought_process', '')
        target_text = row['target_text']
        
        combined_text = (
            f"{input_text} <think>{thought_process}</think> "
            f"<answer>{target_text}</answer>"
        )
        
        input_ids = self.tokenizer.encode(
            combined_text,
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )
        
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
        
        return {
            'input_ids': input_ids.squeeze(0),
            'attention_mask': attention_mask.squeeze(0),
            'labels': input_ids.squeeze(0)  # Use same sequence for labels in CoT
        }
        
    def _process_tot(self, row):
        """Process row for tree-of-thoughts training"""
        input_text = row['input_text']
        thought_tree = row.get('thought_tree', [])
        target_text = row['target_text']
        
        # Format thought tree into a string
        thought_sequence = " -> ".join(thought_tree)
        
        combined_text = (
            f"{input_text} <think>{thought_sequence}</think> "
            f"<answer>{target_text}</answer>"
        )
        
        input_ids = self.tokenizer.encode(
            combined_text,
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )
        
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
        
        return {
            'input_ids': input_ids.squeeze(0),
            'attention_mask': attention_mask.squeeze(0),
            'labels': input_ids.squeeze(0)  # Use same sequence for labels in ToT
        }
        
    def cleanup(self):
        """Cleanup resources"""
        if self.smallpond_manager:
            self.smallpond_manager.cleanup()

def create_dataloader(
    dataset: Union[VishwamAIDataset, pd.DataFrame],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    sampler = None
) -> DataLoader:
    """
    Create a DataLoader with appropriate settings.
    
    Args:
        dataset: Dataset or DataFrame to load
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        sampler: Optional sampler for distributed training
        
    Returns:
        DataLoader instance
    """
    if isinstance(dataset, pd.DataFrame):
        # Convert DataFrame to VishwamAIDataset if needed
        dataset = VishwamAIDataset(
            data_path=dataset,
            tokenizer=None,  # Must be provided when converting
            mode='normal'
        )
        
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=sampler
    )