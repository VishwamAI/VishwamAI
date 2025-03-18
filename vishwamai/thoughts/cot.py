"""
Chain-of-thought implementation for VishwamAI transformer.
"""

import jax
import jax.numpy as jnp
from typing import Any, Dict, List, Optional, Tuple

def format_cot_prompt(question: str, examples: List[Dict[str, str]] = None) -> str:
    """
    Format a prompt for chain-of-thought reasoning.
    
    Args:
        question: The question to reason about
        examples: Optional few-shot examples with reasoning steps
    """
    prompt = ""
    
    # Add few-shot examples if provided
    if examples:
        for example in examples:
            prompt += f"Question: {example['question']}\n"
            prompt += "Let's approach this step by step:\n"
            prompt += f"{example['reasoning']}\n"
            prompt += f"Therefore, the answer is: {example['answer']}\n\n"
    
    # Add the actual question
    prompt += f"Question: {question}\n"
    prompt += "Let's approach this step by step:\n"
    
    return prompt

def extract_reasoning_steps(
    output: str,
    separator: str = "Therefore, the answer is:"
) -> Tuple[List[str], str]:
    """
    Extract reasoning steps and final answer from model output.
    
    Args:
        output: Raw model output text
        separator: String separating reasoning from answer
    """
    # Split reasoning and answer
    parts = output.split(separator)
    if len(parts) != 2:
        return [], output.strip()
    
    reasoning, answer = parts
    
    # Extract steps
    steps = [step.strip() for step in reasoning.split('\n') if step.strip()]
    
    return steps, answer.strip()

def cot_generate(
    model: Any,
    tokenizer: Any,
    question: str,
    examples: Optional[List[Dict[str, str]]] = None,
    max_length: int = 512,
    temperature: float = 0.7,
    num_return_sequences: int = 1,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Generate chain-of-thought reasoning and answer.
    
    Args:
        model: The transformer model
        tokenizer: Tokenizer for text encoding/decoding
        question: Input question
        examples: Optional few-shot examples
        max_length: Maximum sequence length
        temperature: Sampling temperature
        num_return_sequences: Number of reasoning paths to generate
    """
    # Format prompt with CoT structure
    prompt = format_cot_prompt(question, examples)
    
    # Generate multiple reasoning paths
    outputs = []
    for _ in range(num_return_sequences):
        # Generate text with model
        output = model.generate(
            tokenizer.encode(prompt),
            max_length=max_length,
            temperature=temperature,
            **kwargs
        )
        decoded = tokenizer.decode(output[0])
        
        # Extract reasoning and answer
        steps, answer = extract_reasoning_steps(decoded)
        
        outputs.append({
            'reasoning_steps': steps,
            'answer': answer,
            'full_output': decoded
        })
    
    return outputs

def evaluate_reasoning_quality(
    steps: List[str],
    criteria: Dict[str, float]
) -> float:
    """
    Evaluate the quality of reasoning steps.
    
    Args:
        steps: List of reasoning steps
        criteria: Dictionary of criteria and their weights
    """
    score = 0.0
    total_weight = sum(criteria.values())
    
    # Length of reasoning chain
    if 'num_steps' in criteria:
        optimal_steps = 3  # Can be adjusted
        step_score = max(0, 1 - abs(len(steps) - optimal_steps) / optimal_steps)
        score += criteria['num_steps'] * step_score
    
    # Logical flow between steps
    if 'logical_flow' in criteria:
        flow_score = 0
        for i in range(len(steps) - 1):
            # Add simple heuristics for logical connections
            if any(connector in steps[i+1].lower() for connector in 
                  ['therefore', 'because', 'so', 'thus', 'hence']):
                flow_score += 1
        flow_score = flow_score / max(1, len(steps) - 1)
        score += criteria['logical_flow'] * flow_score
    
    # Normalize final score
    return score / total_weight

def cot_inference_step(
    params: Any,
    batch: Dict[str, jnp.ndarray],
    model: Any,
    temperature: float = 1.0
) -> Dict[str, jnp.ndarray]:
    """
    Inference step for chain-of-thought reasoning.
    
    Args:
        params: Model parameters
        batch: Batch of input data
        model: The transformer model
        temperature: Sampling temperature
    """
    # Get model predictions
    logits = model.apply(
        {'params': params},
        batch['input_ids'],
        deterministic=True
    )
    
    # Temperature scaling
    scaled_logits = logits / temperature
    
    # Sample from logits
    predictions = jax.random.categorical(
        jax.random.PRNGKey(0),
        scaled_logits
    )
    
    return {
        'logits': logits,
        'predictions': predictions
    }

class ChainOfThoughtPrompting:
    """Helper class for chain-of-thought prompting."""
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_length: int = 512
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.few_shot_examples = few_shot_examples
        self.temperature = temperature
        self.max_length = max_length
    
    def generate_reasoning(
        self,
        question: str,
        num_paths: int = 1,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate multiple reasoning paths for a question."""
        return cot_generate(
            self.model,
            self.tokenizer,
            question,
            self.few_shot_examples,
            self.max_length,
            self.temperature,
            num_return_sequences=num_paths,
            **kwargs
        )
    
    def evaluate_reasoning(
        self,
        reasoning_outputs: List[Dict[str, Any]],
        criteria: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """Evaluate and rank different reasoning paths."""
        if criteria is None:
            criteria = {
                'num_steps': 0.4,
                'logical_flow': 0.6
            }
        
        # Evaluate each reasoning path
        for output in reasoning_outputs:
            quality_score = evaluate_reasoning_quality(
                output['reasoning_steps'],
                criteria
            )
            output['quality_score'] = quality_score
        
        # Sort by quality score
        return sorted(
            reasoning_outputs,
            key=lambda x: x['quality_score'],
            reverse=True
        )
    
    def reason(
        self,
        question: str,
        num_paths: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate and evaluate multiple reasoning paths,
        returning the best one.
        """
        # Generate multiple reasoning paths
        reasoning_outputs = self.generate_reasoning(
            question,
            num_paths=num_paths,
            **kwargs
        )
        
        # Evaluate and rank paths
        ranked_outputs = self.evaluate_reasoning(reasoning_outputs)
        
        # Return the best reasoning path
        return ranked_outputs[0]