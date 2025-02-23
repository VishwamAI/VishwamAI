"""Text preprocessing module for VishwamAI."""

import re
import unicodedata
from typing import List, Dict, Optional, Union
import random
from pathlib import Path
import yaml
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Handles text preprocessing including cleaning, normalization, and augmentation."""
    
    def __init__(self, config_path: Union[str, Path]):
        """Initialize preprocessor with configuration.
        
        Args:
            config_path: Path to data_config.yaml
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)["preprocessing"]
        
        self.max_length = self.config["max_sequence_length"]
        self.cleaning_config = self.config["text_cleaning"]
        self.augmentation_config = self.config["augmentation"]
        
    def clean_text(self, text: str) -> str:
        """Clean text by removing unwanted elements and normalizing.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Remove HTML if configured
        if self.cleaning_config["remove_html"]:
            text = re.sub(r'<[^>]+>', '', text)
            
        # Normalize Unicode if configured
        if self.cleaning_config["normalize_unicode"]:
            text = unicodedata.normalize('NFKC', text)
            
        # Normalize whitespace if configured
        if self.cleaning_config["normalize_whitespace"]:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
        # Fix common typos if configured
        if self.cleaning_config["fix_common_typos"]:
            # Add common typo fixes here
            text = re.sub(r'(\w)\.(\w)', r'\1. \2', text)  # Fix missing spaces after periods
            text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Fix missing spaces between sentences
            
        return text
        
    def truncate_text(self, text: str) -> str:
        """Truncate text to maximum sequence length.
        
        Args:
            text: Input text to truncate
            
        Returns:
            Truncated text
        """
        words = text.split()
        if len(words) > self.max_length:
            return ' '.join(words[:self.max_length])
        return text
        
    def augment_text(self, text: str) -> List[str]:
        """Apply text augmentation techniques.
        
        Args:
            text: Input text to augment
            
        Returns:
            List of augmented text versions
        """
        if not self.augmentation_config["enabled"]:
            return [text]
            
        augmented_texts = [text]
        words = text.split()
        
        if len(words) < 2:  # Skip short texts
            return augmented_texts
            
        num_augmentations = random.randint(1, self.augmentation_config["max_augmentations_per_example"])
        
        for _ in range(num_augmentations):
            if random.random() > self.augmentation_config["probability"]:
                continue
                
            technique = random.choice(self.augmentation_config["techniques"])
            
            if technique == "random_insertion":
                pos = random.randint(0, len(words))
                word_to_insert = random.choice(words)
                augmented = words.copy()
                augmented.insert(pos, word_to_insert)
                
            elif technique == "random_deletion":
                if len(words) > 3:  # Ensure we don't delete too much
                    pos = random.randint(0, len(words) - 1)
                    augmented = words.copy()
                    del augmented[pos]
                else:
                    continue
                    
            elif technique == "random_swap":
                if len(words) > 1:
                    pos1, pos2 = random.sample(range(len(words)), 2)
                    augmented = words.copy()
                    augmented[pos1], augmented[pos2] = augmented[pos2], augmented[pos1]
                else:
                    continue
                    
            augmented_texts.append(' '.join(augmented))
            
        return augmented_texts
        
    def preprocess_batch(self, texts: List[str], 
                        num_threads: Optional[int] = None) -> List[str]:
        """Preprocess a batch of texts in parallel.
        
        Args:
            texts: List of input texts
            num_threads: Number of threads for parallel processing
            
        Returns:
            List of preprocessed texts
        """
        if num_threads is None:
            num_threads = min(len(texts), 8)  # Default to 8 threads max
            
        processed_texts = []
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Process cleaning and truncation in parallel
            clean_and_truncate = partial(self._clean_and_truncate)
            processed_texts.extend(executor.map(clean_and_truncate, texts))
            
            # Process augmentation if enabled
            if self.augmentation_config["enabled"]:
                augmented_texts = []
                for text in processed_texts:
                    augmented_texts.extend(self.augment_text(text))
                processed_texts = augmented_texts
                
        return processed_texts
        
    def _clean_and_truncate(self, text: str) -> str:
        """Helper method to combine cleaning and truncation.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned and truncated text
        """
        return self.truncate_text(self.clean_text(text))
        
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess a single text input.
        
        Args:
            text: Input text
            
        Returns:
            List of preprocessed text versions (including augmentations)
        """
        cleaned_text = self.clean_text(text)
        truncated_text = self.truncate_text(cleaned_text)
        return self.augment_text(truncated_text)
        
    def get_config(self) -> Dict:
        """Get current preprocessing configuration.
        
        Returns:
            Dictionary containing preprocessing configuration
        """
        return self.config
