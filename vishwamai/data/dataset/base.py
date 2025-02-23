"""Base dataset class for all implementations."""
from typing import Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, data_path: str, tokenizer_path: Optional[str] = None):
        """Initialize base dataset.
        
        Args:
            data_path: Path to dataset files
            tokenizer_path: Optional path to tokenizer
        """
        super().__init__()
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        
    def __len__(self) -> int:
        """Return length of dataset."""
        raise NotImplementedError
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item from dataset.
        
        Args:
            idx: Index to retrieve
            
        Returns:
            Dict containing tokenized inputs and labels
        """
        raise NotImplementedError
        
    def prepare_sample(self, text: str) -> Dict[str, torch.Tensor]:
        """Prepare a single sample.
        
        Args:
            text: Input text to process
            
        Returns:
            Dict containing processed and tokenized input
        """
        raise NotImplementedError
        
    def collate_fn(self, samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate samples into batches.
        
        Args:
            samples: List of individual samples
            
        Returns:
            Dict containing batched inputs
        """
        raise NotImplementedError
