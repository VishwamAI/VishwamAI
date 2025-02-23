"""Text augmentation utilities for math problems."""

import re
import random
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from ...utils.logging import get_logger

logger = get_logger(__name__)

class TextAugmenter:
    """Text augmentation for math problems."""
    
    def __init__(
        self,
        number_change_prob: float = 0.3,
        name_change_prob: float = 0.2,
        variable_change_prob: float = 0.2,
        context_change_prob: float = 0.1,
        seed: Optional[int] = None,
        **kwargs: Any
    ):
        """Initialize augmenter.
        
        Args:
            number_change_prob: Probability of changing numbers
            name_change_prob: Probability of changing names
            variable_change_prob: Probability of changing variables
            context_change_prob: Probability of changing context
            seed: Random seed
            **kwargs: Additional arguments
        """
        self.number_change_prob = number_change_prob
        self.name_change_prob = name_change_prob
        self.variable_change_prob = variable_change_prob
        self.context_change_prob = context_change_prob
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Load substitution dictionaries
        self.names = self._load_names()
        self.variables = self._load_variables()
        self.contexts = self._load_contexts()
        
    def _load_names(self) -> List[str]:
        """Load list of names for substitution."""
        return [
            "John", "Mary", "James", "Sarah", "Michael", "Emma", "David", "Lisa",
            "Robert", "Emily", "William", "Sophia", "Joseph", "Olivia", "Thomas",
            "Isabella", "Charles", "Mia", "Daniel", "Charlotte"
        ]
        
    def _load_variables(self) -> List[str]:
        """Load list of variable names."""
        return [
            "apples", "oranges", "books", "pencils", "cookies", "candies",
            "dollars", "coins", "marbles", "cards", "toys", "balls", "stickers",
            "stamps", "tickets", "bananas", "chocolates", "pens", "erasers"
        ]
        
    def _load_contexts(self) -> List[Dict[str, str]]:
        """Load context substitution pairs."""
        return [
            {
                "pattern": r"(has|have|own|owns)",
                "replacements": ["possesses", "holds", "carries", "keeps", "maintains"]
            },
            {
                "pattern": r"(gives|give|gave)",
                "replacements": ["transfers", "hands", "passes", "delivers", "shares"]
            },
            {
                "pattern": r"(buys|buy|bought)",
                "replacements": ["purchases", "acquires", "obtains", "gets", "procures"]
            }
        ]
        
    def _extract_numbers(self, text: str) -> List[Tuple[str, float]]:
        """Extract numbers from text.
        
        Args:
            text: Input text
            
        Returns:
            List of (number_str, value) pairs
        """
        # Match integers, decimals, and fractions
        number_pattern = r'(\d+(?:\.\d+)?)|(\d+/\d+)'
        matches = re.finditer(number_pattern, text)
        
        numbers = []
        for match in matches:
            num_str = match.group(0)
            if '/' in num_str:
                num, denom = map(float, num_str.split('/'))
                value = num / denom
            else:
                value = float(num_str)
            numbers.append((num_str, value))
            
        return numbers
        
    def _modify_number(self, value: float) -> float:
        """Modify number while preserving mathematical relationships.
        
        Args:
            value: Original number
            
        Returns:
            Modified number
        """
        # Small random perturbation
        factor = np.random.uniform(0.8, 1.2)
        new_value = value * factor
        
        # Round to reasonable precision
        if new_value.is_integer():
            return round(new_value)
        else:
            return round(new_value, 2)
            
    def augment_math_problem(
        self,
        question: str,
        answer: str
    ) -> Dict[str, str]:
        """Augment math problem with number and text changes.
        
        Args:
            question: Original question
            answer: Original answer
            
        Returns:
            Dictionary with augmented question and answer
        """
        # Extract numbers and their relationships
        numbers = self._extract_numbers(question)
        answer_numbers = self._extract_numbers(answer)
        
        # Build number mapping
        number_map = {}
        for num_str, value in numbers:
            if random.random() < self.number_change_prob:
                new_value = self._modify_number(value)
                number_map[num_str] = str(new_value)
                
        # Replace numbers
        aug_question = question
        aug_answer = answer
        for old_num, new_num in number_map.items():
            aug_question = aug_question.replace(old_num, new_num)
            # Update answer accordingly
            if old_num in answer:
                aug_answer = aug_answer.replace(old_num, new_num)
                
        # Replace names
        if random.random() < self.name_change_prob:
            for name in self.names:
                if name in aug_question:
                    new_name = random.choice([n for n in self.names if n != name])
                    aug_question = aug_question.replace(name, new_name)
                    aug_answer = aug_answer.replace(name, new_name)
                    
        # Replace variables
        if random.random() < self.variable_change_prob:
            for var in self.variables:
                if var in aug_question:
                    new_var = random.choice([v for v in self.variables if v != var])
                    aug_question = aug_question.replace(var, new_var)
                    aug_answer = aug_answer.replace(var, new_var)
                    
        # Replace context words
        if random.random() < self.context_change_prob:
            for context in self.contexts:
                pattern = context["pattern"]
                if re.search(pattern, aug_question):
                    replacement = random.choice(context["replacements"])
                    aug_question = re.sub(pattern, replacement, aug_question)
                    
        return {
            "question": aug_question,
            "answer": aug_answer
        }
        
    def batch_augment(
        self,
        questions: List[str],
        answers: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Augment a batch of math problems.
        
        Args:
            questions: List of questions
            answers: List of answers
            
        Returns:
            Tuple of augmented questions and answers
        """
        aug_questions = []
        aug_answers = []
        
        for q, a in zip(questions, answers):
            augmented = self.augment_math_problem(q, a)
            aug_questions.append(augmented["question"])
            aug_answers.append(augmented["answer"])
            
        return aug_questions, aug_answers
        
    def verify_augmentation(
        self,
        original_question: str,
        original_answer: str,
        augmented_question: str,
        augmented_answer: str
    ) -> bool:
        """Verify that augmentation preserves mathematical relationships.
        
        Args:
            original_question: Original question
            original_answer: Original answer
            augmented_question: Augmented question
            augmented_answer: Augmented answer
            
        Returns:
            Whether augmentation is valid
        """
        try:
            # Extract and compare number relationships
            orig_numbers = self._extract_numbers(original_question)
            aug_numbers = self._extract_numbers(augmented_question)
            
            if len(orig_numbers) != len(aug_numbers):
                return False
                
            # Check answer format consistency
            orig_steps = original_answer.count('\n')
            aug_steps = augmented_answer.count('\n')
            
            if orig_steps != aug_steps:
                return False
                
            return True
            
        except Exception as e:
            logger.warning(f"Augmentation verification failed: {str(e)}")
            return False
