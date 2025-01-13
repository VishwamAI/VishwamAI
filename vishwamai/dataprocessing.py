import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Any
import json
import numpy as np
from dataclasses import dataclass
import random

@dataclass
class DataCollatorForLanguageModeling:
    tokenizer: Any
    mlm: bool = False
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    
    def __post_init__(self):
        if self.mlm and not hasattr(self.tokenizer, "mask_token_id"):
            raise ValueError("Tokenizer must have a mask_token_id for MLM")
            
    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Handling variable length sequences
        max_length = max(len(example["input_ids"]) for example in examples)
        
        # Pad to multiple of N if specified
        if self.pad_to_multiple_of is not None:
            max_length = (
                (max_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )
        
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        
        for example in examples:
            input_length = len(example["input_ids"])
            padding_length = max_length - input_length
            
            # Pad input_ids and attention_mask
            input_ids = example["input_ids"] + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = [1] * input_length + [0] * padding_length
            
            if self.mlm:
                labels = self._mask_tokens(torch.tensor(input_ids))
            else:
                labels = input_ids.copy()
                labels = [-100 if token == self.tokenizer.pad_token_id else token for token in labels]
            
            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)
        
        # Convert to tensors
        batch = {k: torch.tensor(v) for k, v in batch.items()}
        
        return batch
    
    def _mask_tokens(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Prepare masked tokens inputs/labels for masked language modeling.
        """
        labels = inputs.clone()
        
        # Sample tokens to mask
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val.tolist()) for val in labels
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # 80% of the time, replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id
        
        # 10% of the time, replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]
        
        # The rest of the time (10%), keep the masked input tokens unchanged
        return labels

class VishwamaiDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_length: int = 2048,
        preprocessing_num_workers: int = 2,  # Reduced if resources are limited
        overwrite_cache: bool = False,
        cache_dir: Optional[str] = None
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = cache_dir
        
        # Load and preprocess data
        self.examples = self.load_and_preprocess_data(
            preprocessing_num_workers,
            overwrite_cache
        )
    
    def load_and_preprocess_data(self, num_workers: int, overwrite_cache: bool):
        # Implement efficient data loading with multiprocessing
        from multiprocessing import Pool
        
        def preprocess(example):
            # Tokenize and truncate/pad sequences
            tokens = self.tokenizer.encode(example['text'])[:self.max_length]
            return {
                "input_ids": tokens,
                "labels": tokens.copy()
            }
        
        with Pool(processes=num_workers) as pool:
            processed = list(pool.map(preprocess, self.load_raw_data()))
        
        # Detect and handle outliers
        processed = self.detect_and_handle_outliers(processed)
        
        return processed
    
    def load_raw_data(self):
        # Implement data loading from data_path
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['examples']
    
    def preprocess_data(data, dtype):
        # Example preprocessing step
        processed_data = data.float().type(dtype)
        return processed_data
    
    def detect_and_handle_outliers(self, data):
        """
        Detect and handle outliers in the dataset.
        """
        lengths = [len(example["input_ids"]) for example in data]
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        
        # Define outlier threshold
        threshold = mean_length + 3 * std_length
        
        # Filter out outliers
        filtered_data = [example for example in data if len(example["input_ids"]) <= threshold]
        
        return filtered_data
    
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        return self.examples[idx]
