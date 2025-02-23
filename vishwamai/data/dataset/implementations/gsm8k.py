"""GSM8K (Grade School Math 8K) dataset implementation."""

from typing import Dict, Any, Optional, Callable
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer

from ..base import BaseDataset
from ...augmentation.text_augment import TextAugmenter
from ....utils.logging import get_logger

logger = get_logger(__name__)

class GSM8KDataset(BaseDataset):
    """Dataset implementation for GSM8K math reasoning problems."""

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        format_func: Optional[Callable] = None,
        augment_data: bool = False,
        **kwargs: Any
    ):
        """Initialize GSM8K dataset.

        Args:
            dataset: HuggingFace dataset
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            format_func: Optional function to format examples
            augment_data: Whether to use data augmentation
            **kwargs: Additional arguments
        """
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format_func = format_func or self.default_format
        self.augment_data = augment_data
        
        if augment_data:
            self.augmenter = TextAugmenter(**kwargs.get("augment_config", {}))
            
    @staticmethod
    def default_format(example: Dict[str, Any]) -> Dict[str, str]:
        """Default formatting for GSM8K examples.
        
        Args:
            example: Dataset example
            
        Returns:
            Formatted example with input and target text
        """
        return {
            "input_text": (
                f"Question: {example['question']}\n"
                "Let's solve this step by step:\n"
            ),
            "target_text": example["answer"]
        }
        
    def _augment_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Apply text augmentation to example.
        
        Args:
            example: Dataset example
            
        Returns:
            Augmented example
        """
        if not self.augment_data:
            return example
            
        # Apply number and text augmentation
        augmented = self.augmenter.augment_math_problem(
            question=example["question"],
            answer=example["answer"]
        )
        return {
            "question": augmented["question"],
            "answer": augmented["answer"]
        }
        
    def _process_example(self, example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process a single example.
        
        Args:
            example: Raw dataset example
            
        Returns:
            Processed example with tensors
        """
        # Apply augmentation
        example = self._augment_example(example)
        
        # Format example
        formatted = self.format_func(example)
        input_text = formatted["input_text"]
        target_text = formatted["target_text"]
        
        # Tokenize input
        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize target
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )["input_ids"]
            
        # Remove batch dimension
        model_inputs = {k: v.squeeze(0) for k, v in model_inputs.items()}
        model_inputs["labels"] = labels.squeeze(0)
        
        return model_inputs
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get processed example by index.
        
        Args:
            idx: Example index
            
        Returns:
            Processed example
        """
        example = self.dataset[idx]
        return self._process_example(example)
        
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.dataset)
        
    def verify_example(self, idx: int) -> bool:
        """Verify that example is properly formatted.
        
        Args:
            idx: Example index
            
        Returns:
            Whether example is valid
        """
        try:
            example = self[idx]
            required_keys = {"input_ids", "attention_mask", "labels"}
            has_required = all(k in example for k in required_keys)
            
            # Check shapes
            correct_shape = all(
                len(example[k].shape) == 1 and
                example[k].shape[0] <= self.max_length
                for k in required_keys
            )
            
            return has_required and correct_shape
            
        except Exception as e:
            logger.warning(f"Example {idx} verification failed: {str(e)}")
            return False
            
    def get_vocab_size(self) -> int:
        """Get vocabulary size from tokenizer."""
        return len(self.tokenizer)
        
    def decode_example(self, example: Dict[str, torch.Tensor]) -> Dict[str, str]:
        """Decode tokenized example back to text.
        
        Args:
            example: Processed example with tensors
            
        Returns:
            Dictionary with decoded input and target text
        """
        input_text = self.tokenizer.decode(
            example["input_ids"],
            skip_special_tokens=True
        )
        target_text = self.tokenizer.decode(
            example["labels"],
            skip_special_tokens=True
        )
        return {
            "input_text": input_text,
            "target_text": target_text
        }
        
    def collate_fn(self, examples: list) -> Dict[str, torch.Tensor]:
        """Collate examples into batches.
        
        Args:
            examples: List of processed examples
            
        Returns:
            Batched examples
        """
        # Stack all tensors
        batch = {
            key: torch.stack([ex[key] for ex in examples])
            for key in examples[0].keys()
        }
        
        return batch
