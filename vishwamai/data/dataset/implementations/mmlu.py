"""MMLU (Massive Multitask Language Understanding) dataset implementation."""
from typing import Dict, List
import json
import torch
from ..base import BaseDataset

class MMLUDataset(BaseDataset):
    def __init__(self, data_path: str, tokenizer_path: str):
        """Initialize MMLU dataset.
        
        Args:
            data_path: Path to MMLU data files
            tokenizer_path: Path to tokenizer
        """
        super().__init__(data_path, tokenizer_path)
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Dict]:
        """Load MMLU samples from data files.
        
        Returns:
            List of samples with questions and answers
        """
        samples = []
        with open(self.data_path, 'r') as f:
            for line in f:
                sample = json.loads(line)
                samples.append({
                    'question': sample['question'],
                    'choices': sample['choices'],
                    'answer': sample['answer']
                })
        return samples
        
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get MMLU sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dict with processed inputs and labels
        """
        sample = self.samples[idx]
        return self.prepare_sample(
            question=sample['question'],
            choices=sample['choices'],
            answer=sample['answer']
        )
        
    def prepare_sample(self, question: str, choices: List[str], answer: int) -> Dict[str, torch.Tensor]:
        """Process MMLU sample into model inputs.
        
        Args:
            question: Question text
            choices: List of answer choices
            answer: Index of correct answer
            
        Returns:
            Dict with tokenized inputs and label
        """
        # Format as single sequence
        text = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            text += f"Choice {i}: {choice}\n"
            
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(answer, dtype=torch.long)
        }
        
    def collate_fn(self, samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate MMLU samples into batch.
        
        Args:
            samples: List of processed samples
            
        Returns:
            Dict with batched tensors
        """
        return {
            'input_ids': torch.stack([s['input_ids'] for s in samples]),
            'attention_mask': torch.stack([s['attention_mask'] for s in samples]), 
            'labels': torch.stack([s['labels'] for s in samples])
        }
