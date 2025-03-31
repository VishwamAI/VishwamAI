import jax
import jax.numpy as jnp
from typing import Any, Dict, List, Optional, Tuple

def format_cot_prompt(question: str, examples: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Format a prompt for chain-of-thought reasoning.

    Args:
        question: The question to reason about.
        examples: Optional few-shot examples with 'question', 'reasoning', and 'answer' keys.

    Returns:
        A formatted prompt string.
    """
    prompt = ""

    # Add few-shot examples if provided
    if examples:
        for example in examples:
            if not all(key in example for key in ['question', 'reasoning', 'answer']):
                raise ValueError("Each example must have 'question', 'reasoning', and 'answer' keys.")
            prompt += f"Question: {example['question']}\n"
            prompt += "Let's approach this step by step:\n"
            prompt += f"{example['reasoning']}\n"
            prompt += f"Therefore, the answer is: {example['answer']}\n\n"

    # Add the actual question
    prompt += f"Question: {question}\n"
    prompt += "Let's approach this step by step:\n"

    return prompt

def extract_reasoning_steps(output: str, separator: str = "Therefore, the answer is:") -> Tuple[List[str], str]:
    """
    Extract reasoning steps and final answer from model output.

    Args:
        output: Raw model output text.
        separator: String separating reasoning from answer.

    Returns:
        A tuple of (reasoning_steps, answer), where reasoning_steps is a list of steps and answer is a string.
    """
    # Split reasoning and answer
    parts = output.split(separator)
    if len(parts) != 2:
        # If separator not found, treat entire output as answer with no steps
        return [], output.strip()

    reasoning, answer = parts

    # Extract steps from reasoning part
    steps = [step.strip() for step in reasoning.split('\n') if step.strip()]

    return steps, answer.strip()

def generate_sequence(
    model: Any,
    params: Any,
    tokenizer: Any,
    input_ids: List[int],
    max_length: int,
    temperature: float,
    key: jax.random.PRNGKey
) -> List[int]:
    """Generates a sequence of tokens autoregressively using a JAX/Flax transformer model.
    
    This function performs token generation by:
    1. Taking the current sequence of tokens
    2. Computing logits from the model
    3. Applying temperature scaling
    4. Sampling the next token
    5. Appending it to the sequence
    6. Repeating until max length or EOS token is reached
    
    Args:
        model: The Flax transformer model for text generation.
        params: Dictionary of model parameters/weights.
        tokenizer: Tokenizer instance for encoding/decoding text.
        input_ids: List of integer token IDs to start generation from.
        max_length: Maximum number of tokens to generate.
        temperature: Sampling temperature - higher values increase diversity.
        key: JAX PRNG key for random sampling.

    Returns:
        List[int]: The generated sequence of token IDs, including the input_ids.
        
    Note:
        - Uses JAX's categorical sampling for token selection
        - Stops early if the model's EOS token is generated
        - Temperature scaling controls randomness in generation
    """
    generated = list(input_ids)
    current_key = key

    while len(generated) < max_length:
        current_key, subkey = jax.random.split(current_key)
        input_array = jnp.array([generated])
        logits = model.apply({'params': params}, input_array, deterministic=True)
        last_logits = logits[:, -1, :] / temperature
        next_token = jax.random.categorical(subkey, last_logits).item()
        generated.append(next_token)

        # Stop if EOS token is generated
        if hasattr(tokenizer, 'eos_token_id') and next_token == tokenizer.eos_token_id:
            break

    return generated

def cot_generate(
    model: Any,
    params: Any,
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
        model: The Flax transformer model.
        params: Model parameters.
        tokenizer: Tokenizer for text encoding/decoding.
        question: Input question.
        examples: Optional few-shot examples.
        max_length: Maximum sequence length.
        temperature: Sampling temperature.
        num_return_sequences: Number of reasoning paths to generate.

    Returns:
        A list of dictionaries containing reasoning steps, answer, and full output.
    """
    prompt = format_cot_prompt(question, examples)
    input_ids = tokenizer.encode(prompt)

    outputs = []
    base_key = jax.random.PRNGKey(0)
    for i in range(num_return_sequences):
        key = jax.random.fold_in(base_key, i)
        generated = generate_sequence(model, params, tokenizer, input_ids, max_length, temperature, key)
        decoded = tokenizer.decode(generated)
        steps, answer = extract_reasoning_steps(decoded)
        outputs.append({
            'reasoning_steps': steps,
            'answer': answer,
            'full_output': decoded
        })

    return outputs

def evaluate_reasoning_quality(steps: List[str], criteria: Dict[str, float]) -> float:
    """
    Evaluate the quality of reasoning steps.

    Args:
        steps: List of reasoning steps.
        criteria: Dictionary of criteria names and their weights (e.g., {'num_steps': 0.4, 'logical_flow': 0.6}).

    Returns:
        A normalized quality score between 0 and 1.
    """
    score = 0.0
    total_weight = sum(criteria.values()) or 1.0  # Avoid division by zero

    # Criterion: Number of steps
    if 'num_steps' in criteria:
        optimal_steps = 3  # Adjustable heuristic
        step_score = max(0, 1 - abs(len(steps) - optimal_steps) / optimal_steps)
        score += criteria['num_steps'] * step_score

    # Criterion: Logical flow
    if 'logical_flow' in criteria:
        flow_score = 0
        if len(steps) > 1:
            for i in range(len(steps) - 1):
                if any(connector in steps[i + 1].lower() for connector in
                       ['therefore', 'because', 'so', 'thus', 'hence']):
                    flow_score += 1
            flow_score /= (len(steps) - 1)
        score += criteria['logical_flow'] * flow_score

    return score / total_weight

class ChainOfThoughtPrompting:
    """Helper class for chain-of-thought prompting."""

    def __init__(
        self,
        model: Any,
        params: Any,
        tokenizer: Any,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_length: int = 512
    ):
        """
        Initialize the CoT prompting class.

        Args:
            model: The Flax transformer model.
            params: Model parameters.
            tokenizer: Tokenizer with encode, decode, and optional eos_token_id.
            few_shot_examples: Optional list of few-shot examples.
            temperature: Sampling temperature.
            max_length: Maximum sequence length.
        """
        self.model = model
        self.params = params
        self.tokenizer = tokenizer
        self.few_shot_examples = few_shot_examples
        self.temperature = temperature
        self.max_length = max_length

    def generate_reasoning(self, question: str, num_paths: int = 1, **kwargs) -> List[Dict[str, Any]]:
        """Generate multiple reasoning paths for a question."""
        return cot_generate(
            self.model,
            self.params,
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
            criteria = {'num_steps': 0.4, 'logical_flow': 0.6}

        # Evaluate each reasoning path
        for output in reasoning_outputs:
            quality_score = evaluate_reasoning_quality(output['reasoning_steps'], criteria)
            output['quality_score'] = quality_score

        # Sort by quality score in descending order
        return sorted(reasoning_outputs, key=lambda x: x['quality_score'], reverse=True)

    def reason(self, question: str, num_paths: int = 3, **kwargs) -> Dict[str, Any]:
        """
        Generate and evaluate multiple reasoning paths, returning the best one.

        Args:
            question: The question to reason about.
            num_paths: Number of reasoning paths to generate and evaluate.

        Returns:
            The highest-scored reasoning path dictionary.
        """
        reasoning_outputs = self.generate_reasoning(question, num_paths=num_paths, **kwargs)
        ranked_outputs = self.evaluate_reasoning(reasoning_outputs)
        return ranked_outputs[0]