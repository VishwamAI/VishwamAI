import os
import logging
from datasets import load_dataset, concatenate_datasets
from typing import Dict, List, Optional
import jax
from functools import partial
from huggingface_hub import HfApi
from tqdm.auto import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HuggingFaceDataPreparator:
    def __init__(
        self,
        model_name: str = "Qwen/QwQ-32B",
        cache_dir: Optional[str] = None,
        use_auth_token: Optional[str] = None
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.use_auth_token = use_auth_token
        self.api = HfApi()

    def prepare_training_data(
        self,
        dataset_names: List[str],
        val_split: float = 0.1,
        max_samples: Optional[int] = None
    ) -> Dict:
        """Prepare training and validation datasets."""
        all_datasets = []
        
        for dataset_name in dataset_names:
            logger.info(f"Loading dataset: {dataset_name}")
            try:
                dataset = load_dataset(
                    dataset_name,
                    use_auth_token=self.use_auth_token,
                    cache_dir=self.cache_dir
                )
                if isinstance(dataset, dict):
                    dataset = dataset['train']
                all_datasets.append(dataset)
            except Exception as e:
                logger.error(f"Error loading {dataset_name}: {str(e)}")
                continue

        if not all_datasets:
            raise ValueError("No datasets were loaded successfully")

        # Combine datasets
        combined = concatenate_datasets(all_datasets)
        
        # Apply max samples limit
        if max_samples:
            combined = combined.select(range(min(max_samples, len(combined))))

        # Split into train/val
        splits = combined.train_test_split(
            test_size=val_split,
            shuffle=True,
            seed=42
        )

        logger.info(f"Created training set with {len(splits['train'])} samples")
        logger.info(f"Created validation set with {len(splits['test'])} samples")

        return splits

    @partial(jax.jit, static_argnums=(0,))
    def preprocess_batch(self, batch, max_length: int = 2048):
        """TPU-optimized batch preprocessing."""
        # Convert to proper dtypes for TPU
        return {
            'input_ids': jax.numpy.asarray(batch['input_ids'], dtype=jax.numpy.int32),
            'attention_mask': jax.numpy.asarray(batch['attention_mask'], dtype=jax.numpy.int32),
            'labels': jax.numpy.asarray(batch['labels'], dtype=jax.numpy.int32)
        }

    def upload_to_hub(
        self,
        processed_dataset,
        repo_name: str,
        private: bool = False
    ):
        """Upload processed dataset to Hugging Face Hub."""
        try:
            processed_dataset.push_to_hub(
                repo_name,
                private=private,
                use_auth_token=self.use_auth_token
            )
            logger.info(f"Successfully uploaded dataset to: {repo_name}")
        except Exception as e:
            logger.error(f"Error uploading to hub: {str(e)}")
            raise

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output-name", type=str, required=True)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--cache-dir", type=str)
    parser.add_argument("--auth-token", type=str)
    
    args = parser.parse_args()

    preparator = HuggingFaceDataPreparator(
        cache_dir=args.cache_dir,
        use_auth_token=args.auth_token
    )

    # Prepare datasets
    splits = preparator.prepare_training_data(
        args.datasets,
        val_split=args.val_split,
        max_samples=args.max_samples
    )

    # Upload to Hub
    preparator.upload_to_hub(
        splits,
        args.output_name,
        private=args.private
    )

if __name__ == "__main__":
    main()
