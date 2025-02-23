"""Text augmentation strategies for data augmentation."""
from typing import List, Optional
import random
import nltk
from nltk.corpus import wordnet
import torch
from transformers import MarianMTModel, MarianTokenizer

class TextAugmenter:
    """Base class for text augmentation strategies."""
    def __init__(self, p: float = 0.1):
        """Initialize augmenter.
        
        Args:
            p: Probability of applying augmentation
        """
        self.p = p
        
    def augment(self, text: str) -> str:
        """Apply augmentation to text.
        
        Args:
            text: Input text to augment
            
        Returns:
            Augmented text
        """
        raise NotImplementedError

class BackTranslation(TextAugmenter):
    """Back translation augmentation using MarianMT models."""
    def __init__(self, p: float = 0.1, target_lang: str = 'fr'):
        """Initialize back translation.
        
        Args:
            p: Probability of applying augmentation
            target_lang: Target language code for intermediate translation
        """
        super().__init__(p)
        self.target_lang = target_lang
        
        # Load translation models
        self.en_to_tgt = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-en-{target_lang}')
        self.tgt_to_en = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{target_lang}-en')
        self.tokenizer = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-en-{target_lang}')
        
    def augment(self, text: str) -> str:
        """Perform back translation augmentation.
        
        Args:
            text: Input text
            
        Returns:
            Augmented text via back translation
        """
        if random.random() > self.p:
            return text
            
        # Translate to target language
        inputs = self.tokenizer(text, return_tensors='pt', padding=True)
        translated = self.en_to_tgt.generate(**inputs)
        tgt_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)
        
        # Translate back to English
        inputs = self.tokenizer(tgt_text, return_tensors='pt', padding=True) 
        translated = self.tgt_to_en.generate(**inputs)
        aug_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)
        
        return aug_text

class SynonymReplacement(TextAugmenter):
    """Replace words with synonyms using WordNet."""
    def __init__(self, p: float = 0.1, num_words: Optional[int] = None):
        """Initialize synonym replacement.
        
        Args:
            p: Probability of applying augmentation per word
            num_words: Number of words to replace, if None use probability
        """
        super().__init__(p)
        self.num_words = num_words
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        
    def _get_synonyms(self, word: str) -> List[str]:
        """Get list of synonyms for a word."""
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() != word:
                    synonyms.append(lemma.name())
        return list(set(synonyms))
        
    def augment(self, text: str) -> str:
        """Replace words with synonyms.
        
        Args:
            text: Input text
            
        Returns:
            Text with words replaced by synonyms
        """
        words = text.split()
        num_words = len(words)
        
        # Determine words to replace
        if self.num_words is not None:
            n_replace = min(self.num_words, num_words)
            replace_idx = random.sample(range(num_words), n_replace)
        else:
            replace_idx = [i for i in range(num_words) if random.random() < self.p]
            
        for i in replace_idx:
            synonyms = self._get_synonyms(words[i])
            if synonyms:
                words[i] = random.choice(synonyms)
                
        return ' '.join(words)

class RandomInsertion(TextAugmenter):
    """Randomly insert synonyms of words in the text."""
    def __init__(self, p: float = 0.1, num_words: Optional[int] = None):
        """Initialize random insertion.
        
        Args:
            p: Probability of inserting after each word
            num_words: Number of words to insert, if None use probability
        """
        super().__init__(p)
        self.num_words = num_words
        self.synonym_replacer = SynonymReplacement(p=1.0)
        
    def augment(self, text: str) -> str:
        """Insert random synonyms into text.
        
        Args:
            text: Input text
            
        Returns:
            Text with inserted words
        """
        words = text.split()
        num_words = len(words)
        
        # Determine number of insertions
        if self.num_words is not None:
            n_insert = min(self.num_words, num_words)
        else:
            n_insert = sum(1 for _ in range(num_words) if random.random() < self.p)
            
        for _ in range(n_insert):
            insert_word = self.synonym_replacer._get_synonyms(random.choice(words))
            if insert_word:
                insert_idx = random.randint(0, len(words))
                words.insert(insert_idx, random.choice(insert_word))
                
        return ' '.join(words)

class RandomSwap(TextAugmenter):
    """Randomly swap words in the text."""
    def __init__(self, p: float = 0.1, num_swaps: Optional[int] = None):
        """Initialize random swapping.
        
        Args:
            p: Probability of swapping each word
            num_swaps: Number of swaps to make, if None use probability
        """
        super().__init__(p)
        self.num_swaps = num_swaps
        
    def augment(self, text: str) -> str:
        """Swap random words in the text.
        
        Args:
            text: Input text
            
        Returns:
            Text with swapped words
        """
        words = text.split()
        num_words = len(words)
        
        if num_words < 2:
            return text
            
        # Determine number of swaps
        if self.num_swaps is not None:
            n_swaps = min(self.num_swaps, num_words // 2)
        else:
            n_swaps = sum(1 for _ in range(num_words) if random.random() < self.p)
            
        for _ in range(n_swaps):
            idx1, idx2 = random.sample(range(num_words), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
            
        return ' '.join(words)

class RandomDeletion(TextAugmenter):
    """Randomly delete words from the text."""
    def __init__(self, p: float = 0.1):
        """Initialize random deletion.
        
        Args:
            p: Probability of deleting each word
        """
        super().__init__(p)
        
    def augment(self, text: str) -> str:
        """Delete random words from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with words deleted
        """
        words = text.split()
        if len(words) == 1:
            return text
            
        new_words = []
        for word in words:
            if random.random() >= self.p:
                new_words.append(word)
                
        if not new_words:  # Ensure at least one word remains
            new_words = [random.choice(words)]
            
        return ' '.join(new_words)
