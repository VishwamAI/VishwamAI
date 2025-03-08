"""
Dataset loading and preparation utilities for VishwamAI training.
Supports CoT, ToT, and standard training data formats.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Union, Tuple
import json
import os
import numpy as np
from tqdm import tqdm

class VishwamAIDataset(Dataset):
    """Base dataset class for VishwamAI training"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512, 
                mode: str = "normal", cache_dir: Optional[str] = None):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to data file/directory
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            mode: Training mode ("normal", "cot", or "tot")
            cache_dir: Directory to cache processed data
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.cache_dir = cache_dir
        
        # Load and process data
        self.examples = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        """Load and preprocess data based on mode"""
        cache_path = os.path.join(self.cache_dir, f"{self.mode}_cache.pt") if self.cache_dir else None
        
        # Try loading from cache first
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached data from {cache_path}")
            return torch.load(cache_path)
        
        # Load raw data
        if self.data_path.endswith('.json'):
            with open(self.data_path, 'r') as f:
                raw_data = json.load(f)
        else:
            raise ValueError(f"Unsupported data format: {self.data_path}")
            
        # Process based on mode
        examples = []
        for item in tqdm(raw_data, desc=f"Processing {self.mode} data"):
            if self.mode == "normal":
                processed = self._process_normal_example(item)
            elif self.mode == "cot":
                processed = self._process_cot_example(item)
            elif self.mode == "tot":
                processed = self._process_tot_example(item)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
                
            if processed:
                examples.append(processed)
                
        # Cache processed data
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(examples, cache_path)
            
        return examples
    
    def _process_normal_example(self, item: Dict) -> Optional[Dict]:
        """Process example for normal training"""
        try:
            input_text = item['input']
            target_text = item['output']
            
            # Tokenize with truncation
            input_ids = self.tokenizer.encode(
                input_text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )[0]
            
            target_ids = self.tokenizer.encode(
                target_text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )[0]
            
            return {
                'input_ids': input_ids,
                'target_ids': target_ids,
                'input_text': input_text,
                'target_text': target_text
            }
        except Exception as e:
            print(f"Error processing example: {e}")
            return None
            
    def _process_cot_example(self, item: Dict) -> Optional[Dict]:
        """Process example for Chain of Thought training"""
        try:
            input_text = item['input']
            thoughts = item.get('thoughts', '')
            answer = item['output']
            
            # Combine thoughts and answer with special tokens
            target_text = f"<think>{thoughts}</think> <answer>{answer}</answer>"
            
            # Tokenize
            input_ids = self.tokenizer.encode(
                input_text,
                max_length=self.max_length // 2,  # Leave room for thoughts
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )[0]
            
            target_ids = self.tokenizer.encode(
                target_text,
                max_length=self.max_length // 2,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )[0]
            
            return {
                'input_ids': input_ids,
                'target_ids': target_ids,
                'input_text': input_text,
                'target_text': target_text,
                'thoughts': thoughts,
                'answer': answer
            }
        except Exception as e:
            print(f"Error processing CoT example: {e}")
            return None
            
    def _process_tot_example(self, item: Dict) -> Optional[Dict]:
        """Process example for Tree of Thoughts training"""
        try:
            input_text = item['input']
            thought_tree = item.get('thought_tree', {})
            answer = item['output']
            
            # Convert thought tree to linear sequence with depth markers
            thought_sequence = self._linearize_thought_tree(thought_tree)
            target_text = f"<think>{thought_sequence}</think> <answer>{answer}</answer>"
            
            # Tokenize
            input_ids = self.tokenizer.encode(
                input_text,
                max_length=self.max_length // 2,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )[0]
            
            target_ids = self.tokenizer.encode(
                target_text,
                max_length=self.max_length // 2,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )[0]
            
            return {
                'input_ids': input_ids,
                'target_ids': target_ids,
                'input_text': input_text,
                'target_text': target_text,
                'thought_tree': thought_tree,
                'answer': answer
            }
        except Exception as e:
            print(f"Error processing ToT example: {e}")
            return None
            
    def _linearize_thought_tree(self, tree: Dict) -> str:
        """Convert a thought tree to a linear sequence with depth markers"""
        def _traverse(node, depth=0):
            if isinstance(node, str):
                return f"[{depth}]{node}"
            
            if isinstance(node, dict):
                thoughts = []
                for key, value in node.items():
                    thoughts.append(f"[{depth}]{key}")
                    if value:  # If there are child thoughts
                        thoughts.extend(_traverse(value, depth + 1).split('\n'))
                return '\n'.join(thoughts)
            
            return ""
            
        return _traverse(tree)
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]

def create_dataloader(
    dataset: VishwamAIDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """Create a DataLoader for training/evaluation"""
    
    def collate_fn(batch):
        # Filter out None values
        batch = [b for b in batch if b is not None]
        if not batch:
            raise ValueError("Empty batch after filtering")
            
        # Stack tensors
        input_ids = torch.stack([example['input_ids'] for example in batch])
        target_ids = torch.stack([example['target_ids'] for example in batch])
        
        # Collect metadata
        metadata = {
            'input_text': [example['input_text'] for example in batch],
            'target_text': [example['target_text'] for example in batch]
        }
        
        # Add mode-specific data
        if 'thoughts' in batch[0]:
            metadata['thoughts'] = [example['thoughts'] for example in batch]
        if 'thought_tree' in batch[0]:
            metadata['thought_tree'] = [example['thought_tree'] for example in batch]
        if 'answer' in batch[0]:
            metadata['answer'] = [example['answer'] for example in batch]
            
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'metadata': metadata
        }
        
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

# Example usage
if __name__ == "__main__":
    # Mock tokenizer for testing
    class MockTokenizer:
        def encode(self, text, max_length=None, truncation=None, padding=None, return_tensors=None):
            if return_tensors == 'pt':
                return torch.randint(0, 1000, (1, max_length))
            return torch.randint(0, 1000, (max_length,))
            
    # Test data
    test_data = [
        {"input": "2+2", "output": "4", "thoughts": "Let me think step by step"},
        {"input": "3+3", "output": "6", "thought_tree": {"First": {"Second": "Third"}}}
    ]
    
    with open("test_data.json", "w") as f:
        json.dump(test_data, f)
        
    # Create dataset instances
    tokenizer = MockTokenizer()
    
    for mode in ["normal", "cot", "tot"]:
        dataset = VishwamAIDataset("test_data.json", tokenizer, mode=mode)
        dataloader = create_dataloader(dataset, batch_size=2)
        
        # Test iteration
        for batch in dataloader:
            print(f"\n{mode} batch:")
            print(f"Input shape: {batch['input_ids'].shape}")
            print(f"Target shape: {batch['target_ids'].shape}")
            print(f"Metadata keys: {batch['metadata'].keys()}")
            break
            
    # Cleanup
    os.remove("test_data.json")