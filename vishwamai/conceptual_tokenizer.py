import torch
import regex as re
import json
from typing import List, Dict, Optional, Union, Set
from collections import defaultdict
from pathlib import Path
import numpy as np

class ConceptNode:
    def __init__(self, value: str = "", score: float = 0.0):
        self.value = value
        self.score = score
        self.children: Dict[str, 'ConceptNode'] = {}
        self.is_concept = False
        self.concept_id: Optional[int] = None

class ConceptualTokenizer:
    def __init__(self, vocab_size: int = 64000, max_length: int = 2048):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.root = ConceptNode()
        self.concept_to_id: Dict[str, int] = {}
        self.id_to_concept: Dict[int, str] = {}
        self.special_tokens = {
            "<pad>": 0,
            "<s>": 1,
            "</s>": 2,
            "<unk>": 3,
            "<mask>": 4,
            "<sep>": 5
        }
        self.concept_patterns = self._compile_patterns()
        self._initialize_special_tokens()
        
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for concept extraction"""
        return {
            'word': re.compile(r'\b\w+\b'),
            'subword': re.compile(r'[A-Z]{2,}(?=[A-Z][a-z]+[0-9]*|\b)|[A-Z]?[a-z]+[0-9]*|[A-Z]|[0-9]+'),
            'punctuation': re.compile(r'[^\w\s]'),
            'whitespace': re.compile(r'\s+')
        }
        
    def _initialize_special_tokens(self):
        """Initialize special tokens in vocabulary"""
        for token, idx in self.special_tokens.items():
            self.concept_to_id[token] = idx
            self.id_to_concept[idx] = token
            
    def train(self, texts: List[str], min_frequency: int = 5):
        """Train tokenizer on a corpus of texts"""
        # Collect concept frequencies
        concept_freq = defaultdict(int)
        for text in texts:
            concepts = self._extract_concepts(text)
            for concept in concepts:
                concept_freq[concept] += 1
                
        # Build concept tree
        for concept, freq in concept_freq.items():
            if freq >= min_frequency:
                self._add_to_tree(concept, freq / len(texts))
                
        # Assign IDs to concepts
        self._assign_concept_ids()
        
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract concepts from text using pattern matching"""
        concepts = []
        
        # Extract words
        words = self.concept_patterns['word'].findall(text)
        concepts.extend(words)
        
        # Extract subwords from compound words
        for word in words:
            if len(word) > 3:  # Only process longer words
                subwords = self.concept_patterns['subword'].findall(word)
                concepts.extend(subwords)
                
        return [c.lower() for c in concepts if c]
        
    def _add_to_tree(self, concept: str, score: float):
        """Add concept to hierarchical tree"""
        current = self.root
        chars = list(concept)
        
        for char in chars:
            if char not in current.children:
                current.children[char] = ConceptNode()
            current = current.children[char]
            
        current.value = concept
        current.score = score
        current.is_concept = True
        
    def _assign_concept_ids(self):
        """Assign IDs to concepts based on scores"""
        concepts = []
        self._collect_concepts(self.root, concepts)
        
        # Sort by score
        concepts.sort(key=lambda x: x[1], reverse=True)
        
        # Assign IDs
        next_id = len(self.special_tokens)
        for concept, _ in concepts:
            if next_id >= self.vocab_size:
                break
            if concept not in self.concept_to_id:
                self.concept_to_id[concept] = next_id
                self.id_to_concept[next_id] = concept
                next_id += 1
                
    def _collect_concepts(self, node: ConceptNode, concepts: List[tuple]):
        """Collect concepts and their scores from tree"""
        if node.is_concept:
            concepts.append((node.value, node.score))
            
        for child in node.children.values():
            self._collect_concepts(child, concepts)
            
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Encode text to concept IDs"""
        if max_length is None:
            max_length = self.max_length
            
        concepts = self._extract_concepts(text)
        concept_ids = []
        
        for concept in concepts:
            if concept in self.concept_to_id:
                concept_ids.append(self.concept_to_id[concept])
            else:
                # Handle unknown concepts
                subparts = self._tokenize_unknown(concept)
                concept_ids.extend(subparts)
                
        # Truncate if necessary
        if len(concept_ids) > max_length - 2:  # Account for special tokens
            concept_ids = concept_ids[:max_length - 2]
            
        # Add special tokens
        concept_ids = [self.special_tokens['<s>']] + concept_ids + [self.special_tokens['</s>']]
        
        return concept_ids
    
    def _tokenize_unknown(self, concept: str) -> List[int]:
        """Handle unknown concepts by breaking into known parts"""
        parts = []
        current_part = ""
        
        for char in concept:
            current_part += char
            if current_part in self.concept_to_id:
                parts.append(self.concept_to_id[current_part])
                current_part = ""
                
        if current_part:
            parts.append(self.special_tokens['<unk>'])
            
        return parts
        
    def decode(self, concept_ids: List[int]) -> str:
        """Decode concept IDs back to text"""
        concepts = []
        for concept_id in concept_ids:
            if concept_id in self.id_to_concept:
                concept = self.id_to_concept[concept_id]
                if concept not in self.special_tokens.keys():
                    concepts.append(concept)
        return " ".join(concepts)
    
    def save(self, path: Union[str, Path]):
        """Save tokenizer to file"""
        save_dict = {
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'concept_to_id': self.concept_to_id,
            'special_tokens': self.special_tokens
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(save_dict, f, ensure_ascii=False, indent=2)
            
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ConceptualTokenizer':
        """Load tokenizer from file"""
        with open(path, 'r', encoding='utf-8') as f:
            save_dict = json.load(f)
            
        tokenizer = cls(
            vocab_size=save_dict['vocab_size'],
            max_length=save_dict['max_length']
        )
        
        tokenizer.concept_to_id = save_dict['concept_to_id']
        tokenizer.id_to_concept = {int(k): v for k, v in tokenizer.concept_to_id.items()}
        tokenizer.special_tokens = save_dict['special_tokens']
        
        return tokenizer

    def encode_with_concepts(
        self,
        text: str,
        max_length: Optional[int] = None,
        max_concepts: Optional[int] = None
    ) -> Dict[str, List[int]]:
        """Encode text and return both token and concept IDs"""
        token_ids = self.encode(text, max_length)
        
        # Extract concepts for each token
        concepts = self._extract_concepts(text)
        concept_ids = []
        
        if max_concepts is None:
            max_concepts = 4  # Default max concepts per token
            
        for concept in concepts:
            if concept in self.concept_to_id:
                concept_ids.append(self.concept_to_id[concept])
            if len(concept_ids) >= max_concepts:
                break
                
        # Pad concept IDs if necessary
        while len(concept_ids) < max_concepts:
            concept_ids.append(self.special_tokens['<pad>'])
            
        return {
            'token_ids': token_ids,
            'concept_ids': concept_ids
        }

    def batch_encode_with_concepts(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        max_concepts: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Batch encode texts with concepts"""
        batch_tokens = []
        batch_concepts = []
        
        for text in texts:
            encoded = self.encode_with_concepts(text, max_length, max_concepts)
            batch_tokens.append(encoded['token_ids'])
            batch_concepts.append(encoded['concept_ids'])
            
        return {
            'token_ids': torch.tensor(batch_tokens),
            'concept_ids': torch.tensor(batch_concepts)
        }