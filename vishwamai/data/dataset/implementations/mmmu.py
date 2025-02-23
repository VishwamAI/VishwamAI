"""MMMU (Multimodal Mathematical Understanding) dataset implementation."""
from typing import Dict, List
import json
import torch
from PIL import Image
from ..base import BaseDataset

class MMMUDataset(BaseDataset):
    def __init__(self, data_path: str, tokenizer_path: str, image_dir: str = None):
        """Initialize MMMU dataset.
        
        Args:
            data_path: Path to MMMU data files
            tokenizer_path: Path to tokenizer
            image_dir: Optional path to directory containing images
        """
        super().__init__(data_path, tokenizer_path)
        self.image_dir = image_dir
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Dict]:
        """Load MMMU samples from data files.
        
        Returns:
            List of samples with questions, context, images and answers
        """
        samples = []
        with open(self.data_path, 'r') as f:
            for line in f:
                sample = json.loads(line)
                samples.append({
                    'question': sample['question'],
                    'context': sample.get('context', ''),
                    'image_path': sample.get('image_path', None),
                    'options': sample.get('options', []),
                    'answer': sample['answer'],
                    'solution': sample.get('solution', '')
                })
        return samples
        
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get MMMU sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dict with processed inputs and labels
        """
        sample = self.samples[idx]
        return self.prepare_sample(
            question=sample['question'],
            context=sample['context'],
            image_path=sample['image_path'],
            options=sample['options'],
            answer=sample['answer'],
            solution=sample['solution']
        )
        
    def prepare_sample(self, question: str, context: str, image_path: str = None,
                      options: List[str] = None, answer: str = None,
                      solution: str = None) -> Dict[str, torch.Tensor]:
        """Process MMMU sample into model inputs.
        
        Args:
            question: Question text
            context: Additional context
            image_path: Optional path to associated image
            options: Optional list of multiple choice options 
            answer: Correct answer text
            solution: Optional solution explanation
            
        Returns:
            Dict with tokenized inputs and labels
        """
        # Combine text inputs
        text = f"Context: {context}\nQuestion: {question}\n"
        if options:
            for i, opt in enumerate(options):
                text += f"Option {i}: {opt}\n"
                
        # Tokenize text
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Load image if present
        image_tensor = None
        if image_path and self.image_dir:
            img_path = f"{self.image_dir}/{image_path}"
            image = Image.open(img_path).convert('RGB')
            # Assume image preprocessing transforms defined elsewhere
            image_tensor = self.image_transform(image)
            
        output = {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
        }
        
        if image_tensor is not None:
            output['image'] = image_tensor
            
        if answer is not None:
            if options:
                # For multiple choice, convert answer to index
                answer_idx = options.index(answer)
                output['labels'] = torch.tensor(answer_idx, dtype=torch.long)
            else:
                # For free-form answers, tokenize the answer text
                answer_tokens = self.tokenizer(
                    answer,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                output['labels'] = answer_tokens['input_ids'].squeeze()
                
        return output
        
    def collate_fn(self, samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate MMMU samples into batch.
        
        Args:
            samples: List of processed samples
            
        Returns:
            Dict with batched tensors
        """
        batch = {
            'input_ids': torch.stack([s['input_ids'] for s in samples]),
            'attention_mask': torch.stack([s['attention_mask'] for s in samples])
        }
        
        # Handle optional tensors
        if 'image' in samples[0]:
            batch['image'] = torch.stack([s['image'] for s in samples])
            
        if 'labels' in samples[0]:
            batch['labels'] = torch.stack([s['labels'] for s in samples])
            
        return batch
