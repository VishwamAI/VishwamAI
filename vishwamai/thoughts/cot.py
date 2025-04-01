"""Chain of Thought implementation for VishwamAI."""

import jax
import jax.numpy as jnp
from typing import Any, Dict, List, Optional, Tuple
import re

def extract_reasoning_steps(text: str) -> Tuple[List[str], str]:
    """Extract reasoning steps and final answer from generated text."""
    # Split into steps and answer
    steps = []
    answer = ""
    
    lines = text.split('\n')
    current_step = []
    
    for line in lines:
        line = line.strip()
        if line.lower().startswith(('step', 'therefore', 'thus', 'so', 'finally')):
            if current_step:
                steps.append(' '.join(current_step))
                current_step = []
            if line.lower().startswith(('therefore', 'thus', 'so', 'finally')):
                answer = line
                break
            current_step.append(line)
        elif current_step or line:
            current_step.append(line)
            
    if current_step and not answer:
        steps.append(' '.join(current_step))
        
    return steps, answer

def validate_reasoning(
    model: Any,
    params: Any,
    tokenizer: Any,
    question: str,
    steps: List[str],
    answer: str,
    key: jax.random.PRNGKey
) -> Tuple[bool, str]:
    """Validate the reasoning steps and answer."""
    validation_prompt = (
        f"Question: {question}\n\n"
        f"Reasoning steps:\n" + "\n".join(steps) + "\n\n"
        f"Answer: {answer}\n\n"
        "Are these reasoning steps logically valid and does the answer follow from them? "
        "First answer YES or NO, then explain why:"
    )
    
    input_ids = tokenizer.encode(
        validation_prompt,
        return_tensors="jax",
        max_length=model.config.max_seq_len,
        truncation=True
    ).tolist()[0]
    
    generated = generate_sequence(
        model=model,
        params=params,
        tokenizer=tokenizer,
        input_ids=input_ids,
        max_length=len(input_ids) + 200,
        temperature=0.7,
        key=key,
        num_return_sequences=1
    )[0]
    
    response = tokenizer.decode(generated, skip_special_tokens=True)
    is_valid = response.lower().startswith('yes')
    
    return is_valid, response

def generate_sequence(
    model: Any,
    params: Any,
    tokenizer: Any,
    input_ids: List[int],
    max_length: int,
    temperature: float,
    key: jax.random.PRNGKey,
    num_return_sequences: int = 1
) -> List[List[int]]:
    """Generate sequences using the model."""
    outputs = []
    
    for i in range(num_return_sequences):
        subkey = jax.random.fold_in(key, i)
        
        # Initialize sequence
        cur_ids = input_ids.copy()
        
        # Auto-regressive generation
        for _ in range(max_length - len(input_ids)):
            logits = model.apply(
                {'params': params},
                jnp.array(cur_ids)[None, :],
                deterministic=True
            )
            
            # Get next token probabilities
            next_token_logits = logits[0, -1, :]
            next_token_logits = next_token_logits / temperature
            next_token_probs = jax.nn.softmax(next_token_logits)
            
            # Sample next token
            next_token = jax.random.categorical(subkey, next_token_logits)
            
            # Break if EOS token
            if next_token == tokenizer.eos_token_id:
                break
                
            cur_ids.append(int(next_token))
            
        outputs.append(cur_ids)
        
    return outputs

class ChainOfThoughtPrompting:
    """Helper class for chain-of-thought prompting."""

    def __init__(
        self,
        model: Any,
        params: Any,
        tokenizer: Any,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_length: int = 512,
        seed: int = 0
    ):
        """Initialize ChainOfThoughtPrompting.
        
        Args:
            model: The language model to use
            params: Model parameters
            tokenizer: Tokenizer for text processing
            few_shot_examples: Optional examples for few-shot prompting
            temperature: Sampling temperature
            max_length: Maximum sequence length
            seed: Random seed
        """
        self.model = model
        self.params = params
        self.tokenizer = tokenizer
        self.few_shot_examples = few_shot_examples or []
        self.temperature = temperature
        self.max_length = max_length
        self.key = jax.random.PRNGKey(seed)
        
    def generate_reasoning(
        self,
        question: str,
        num_paths: int = 1,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate multiple reasoning paths for a question."""
        # Construct prompt with examples if available
        examples_text = ""
        if self.few_shot_examples:
            for ex in self.few_shot_examples:
                examples_text += (
                    f"Question: {ex['question']}\n"
                    f"Let's solve this step by step:\n{ex['reasoning']}\n"
                    f"Therefore, {ex['answer']}\n\n"
                )
                
        prompt = (
            f"{examples_text}Question: {question}\n"
            "Let's solve this step by step:"
        )
        
        input_ids = self.tokenizer.encode(
            prompt,
            return_tensors="jax",
            max_length=self.max_length,
            truncation=True
        ).tolist()[0]
        
        outputs = []
        for _ in range(num_paths):
            self.key, subkey = jax.random.split(self.key)
            
            generated = generate_sequence(
                model=self.model,
                params=self.params,
                tokenizer=self.tokenizer,
                input_ids=input_ids,
                max_length=self.max_length,
                temperature=self.temperature,
                key=subkey,
                num_return_sequences=1
            )[0]
            
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            steps, answer = extract_reasoning_steps(text)
            
            self.key, subkey = jax.random.split(self.key)
            is_valid, validation = validate_reasoning(
                self.model,
                self.params,
                self.tokenizer,
                question,
                steps,
                answer,
                subkey
            )
            
            outputs.append({
                'reasoning_steps': steps,
                'answer': answer,
                'is_valid': is_valid,
                'validation': validation,
                'full_output': text
            })
            
        return outputs
        
    def evaluate_reasoning(
        self,
        reasoning_outputs: List[Dict[str, Any]],
        criteria: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """Evaluate and rank different reasoning paths."""
        criteria = criteria or {
            'validity': 0.4,
            'completeness': 0.3,
            'clarity': 0.3
        }
        
        for output in reasoning_outputs:
            # Score different aspects
            scores = {}
            
            # Validity score from validation
            scores['validity'] = 1.0 if output['is_valid'] else 0.0
            
            # Completeness score based on number of steps
            num_steps = len(output['reasoning_steps'])
            scores['completeness'] = min(1.0, num_steps / 5)  # Cap at 5 steps
            
            # Clarity score based on average step length
            avg_step_len = sum(len(s.split()) for s in output['reasoning_steps']) / max(1, num_steps)
            scores['clarity'] = min(1.0, 20 / avg_step_len)  # Prefer concise steps
            
            # Calculate weighted score
            output['score'] = sum(
                scores[k] * v for k, v in criteria.items()
            )
            output['aspect_scores'] = scores
            
        # Sort by score
        reasoning_outputs.sort(key=lambda x: x['score'], reverse=True)
        return reasoning_outputs
        
    def reason(
        self,
        question: str,
        num_paths: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate and evaluate reasoning paths to answer a question.
        
        Args:
            question: The question to answer
            num_paths: Number of reasoning paths to generate
            **kwargs: Additional arguments for generation
            
        Returns:
            Dictionary containing best reasoning path and metrics
        """
        # Generate multiple reasoning paths
        paths = self.generate_reasoning(
            question,
            num_paths=num_paths,
            **kwargs
        )
        
        # Evaluate paths
        evaluated_paths = self.evaluate_reasoning(paths)
        
        # Return best path and stats
        return {
            'question': question,
            'best_reasoning': evaluated_paths[0],
            'all_paths': evaluated_paths,
            'stats': {
                'num_paths': len(paths),
                'avg_score': sum(p['score'] for p in evaluated_paths) / len(evaluated_paths),
                'max_score': evaluated_paths[0]['score']
            }
        }