"""GSM8K (Grade School Math 8K) dataset implementation."""
from typing import Dict, List
import json
import torch
from ..base import BaseDataset

class GSM8KDataset(BaseDataset):
    def __init__(self, data_path: str, tokenizer_path: str):
        """Initialize GSM8K dataset.
        
        Args:
            data_path: Path to GSM8K data files
            tokenizer_path: Path to tokenizer
        """
        super().__init__(data_path, tokenizer_path)
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Dict]:
        """Load GSM8K samples from data files.
        
        Returns:
            List of samples with questions and answers
        """
        samples = []
        with open(self.data_path, 'r') as f:
            for line in f:
                sample = json.loads(line)
                samples.append({
                    'question': sample['question'],
                    'answer': sample['answer'],
                    'solution': sample.get('solution', ''),
                })
        return samples
        
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get GSM8K sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dict with processed inputs and labels
        """
        sample = self.samples[idx]
        return self.prepare_sample(
            question=sample['question'],
            answer=sample['answer'],
            solution=sample['solution']
        )
        
    def prepare_sample(self, question: str, answer: str,
                      solution: str = '') -> Dict[str, torch.Tensor]:
        """Process GSM8K sample into model inputs.
        
        Args:
            question: Question text
            answer: Answer text (typically numerical)
            solution: Optional step-by-step solution
            
        Returns:
            Dict with tokenized inputs and label
        """
        # Format input text
        if solution:
            text = f"Question: {question}\nSolution: {solution}\nAnswer: {answer}"
        else:
            text = f"Question: {question}\nAnswer: {answer}"
            
        # Tokenize input
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # For training, we use the answer text as the target
        answer_tokens = self.tokenizer(
            str(answer),  # Ensure answer is string
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': answer_tokens['input_ids'].squeeze()
        }
        
    def collate_fn(self, samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate GSM8K samples into batch.
        
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
