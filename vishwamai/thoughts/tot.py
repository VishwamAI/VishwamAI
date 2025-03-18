"""Tree of Thoughts implementation for VishwamAI."""
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, List, Optional, Tuple
import random

from dataclasses import dataclass

@dataclass
class ThoughtNode:
    """Represents a node in the tree of thoughts."""
    thought: str
    value: float
    children: List['ThoughtNode'] = None
    parent: Optional['ThoughtNode'] = None
    depth: int = 0

    def __post_init__(self):
        if self.children is None:
            self.children = []

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
    """
    Generate sequences of tokens autoregressively using the model.

    Args:
        model: JAX/Flax model instance.
        params: Model parameters.
        tokenizer: Tokenizer instance.
        input_ids: Initial input token IDs.
        max_length: Maximum sequence length (including input).
        temperature: Sampling temperature.
        key: JAX random key for sampling.
        num_return_sequences: Number of sequences to generate.

    Returns:
        A list of generated token ID sequences.
    """
    generated = [list(input_ids) for _ in range(num_return_sequences)]
    current_key = key

    for _ in range(max_length - len(input_ids)):
        new_generated = []
        for seq in generated:
            current_key, subkey = jax.random.split(current_key)
            input_array = jnp.array([seq])
            logits = model.apply({'params': params}, input_array, deterministic=True)
            last_logits = logits[:, -1, :] / temperature
            next_token = jax.random.categorical(subkey, last_logits).item()
            new_seq = seq + [next_token]
            new_generated.append(new_seq)

            # Stop if EOS token is generated
            if hasattr(tokenizer, 'eos_token_id') and next_token == tokenizer.eos_token_id:
                continue

        generated = new_generated

    return generated

class TreeOfThoughts:
    """Implements tree-based reasoning for complex problem solving."""

    def __init__(
        self,
        model: Any,
        params: Any,
        tokenizer: Any,
        max_branches: int = 3,
        max_depth: int = 3,
        beam_width: int = 5,
        temperature: float = 0.7,
        seed: int = 0
    ):
        """
        Initialize the TreeOfThoughts instance.

        Args:
            model: JAX/Flax model instance.
            params: Model parameters.
            tokenizer: Tokenizer instance.
            max_branches: Maximum number of child nodes per expansion.
            max_depth: Maximum depth of the thought tree.
            beam_width: Number of top nodes to keep during search.
            temperature: Default sampling temperature.
            seed: Random seed for reproducibility.
        """
        self.model = model
        self.params = params
        self.tokenizer = tokenizer
        self.max_branches = max_branches
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.temperature = temperature
        self.key = jax.random.PRNGKey(seed)

    def generate_thoughts(
        self,
        context: str,
        num_thoughts: int = 3,
        temperature: Optional[float] = None
    ) -> List[str]:
        """
        Generate multiple potential thought branches.

        Args:
            context: Current context/prompt.
            num_thoughts: Number of thoughts to generate.
            temperature: Optional override for sampling temperature.

        Returns:
            List of generated thoughts.
        """
        temperature = temperature or self.temperature

        # Prepare input
        input_ids = self.tokenizer.encode(
            context,
            return_tensors="jax",
            max_length=self.model.config.max_seq_len,
            truncation=True
        ).tolist()[0]

        thoughts = []
        for _ in range(num_thoughts):
            self.key, subkey = jax.random.split(self.key)
            generated = generate_sequence(
                model=self.model,
                params=self.params,
                tokenizer=self.tokenizer,
                input_ids=input_ids,
                max_length=self.model.config.max_seq_len,
                temperature=temperature,
                key=subkey,
                num_return_sequences=1
            )[0]
            thought = self.tokenizer.decode(generated, skip_special_tokens=True)
            thoughts.append(thought.strip())

        return thoughts

    def estimate_value(
        self,
        thought_sequence: List[str],
        objective: str
    ) -> float:
        """
        Estimate the value/quality of a thought sequence.

        Args:
            thought_sequence: Sequence of thoughts to evaluate.
            objective: Goal/objective for evaluation.

        Returns:
            Estimated value between 0 and 1.
        """
        eval_prompt = (
            f"Objective: {objective}\n\n"
            f"Thought sequence:\n"
            + "\n".join(f"{i+1}. {t}" for i, t in enumerate(thought_sequence))
            + "\n\nOn a scale of 0 to 1, rate how well this thought sequence achieves the objective."
        )

        input_ids = self.tokenizer.encode(
            eval_prompt,
            return_tensors="jax",
            max_length=self.model.config.max_seq_len,
            truncation=True
        ).tolist()[0]

        self.key, subkey = jax.random.split(self.key)
        generated = generate_sequence(
            model=self.model,
            params=self.params,
            tokenizer=self.tokenizer,
            input_ids=input_ids,
            max_length=len(input_ids) + 20,
            temperature=0.1,
            key=subkey,
            num_return_sequences=1
        )[0]

        response = self.tokenizer.decode(generated, skip_special_tokens=True)

        try:
            value = float(next(
                float(s) for s in response.split() if s.replace(".", "").isdigit()
            ))
            return max(0.0, min(1.0, value))
        except (StopIteration, ValueError):
            return 0.0

    def expand_node(
        self,
        node: ThoughtNode,
        context: str,
        objective: str
    ) -> List[ThoughtNode]:
        """
        Expand a node by generating child thoughts.

        Args:
            node: Node to expand.
            context: Current context.
            objective: Goal/objective.

        Returns:
            List of child nodes.
        """
        if node.depth >= self.max_depth:
            return []

        expanded_context = (
            f"{context}\n"
            f"Current thought: {node.thought}\n"
            f"Given this, generate the next step:"
        )

        new_thoughts = self.generate_thoughts(
            expanded_context,
            num_thoughts=self.max_branches
        )

        children = []
        for thought in new_thoughts:
            thought_sequence = self._get_thought_sequence(node) + [thought]
            value = self.estimate_value(thought_sequence, objective)
            child = ThoughtNode(
                thought=thought,
                value=value,
                parent=node,
                depth=node.depth + 1
            )
            children.append(child)

        node.children = children
        return children

    def _get_thought_sequence(self, node: ThoughtNode) -> List[str]:
        """
        Get the sequence of thoughts from root to node.

        Args:
            node: Target node.

        Returns:
            List of thoughts in the sequence.
        """
        sequence = []
        current = node
        while current:
            sequence.insert(0, current.thought)
            current = current.parent
        return sequence

    def search(
        self,
        initial_prompt: str,
        objective: str,
        max_steps: int = 10
    ) -> List[str]:
        """
        Perform tree search to find solution path.

        Args:
            initial_prompt: Starting prompt/context.
            objective: Goal/objective.
            max_steps: Maximum search steps.

        Returns:
            Best thought sequence found.
        """
        thoughts = self.generate_thoughts(initial_prompt)

        nodes = [
            ThoughtNode(
                thought=t,
                value=self.estimate_value([t], objective),
                parent=None,
                depth=0
            )
            for t in thoughts
        ]

        best_sequence = None
        best_value = 0.0

        for _ in range(max_steps):
            nodes.sort(key=lambda n: n.value, reverse=True)
            nodes = nodes[:self.beam_width]

            if nodes and nodes[0].value > best_value:
                best_value = nodes[0].value
                best_sequence = self._get_thought_sequence(nodes[0])

            new_nodes = []
            for node in nodes:
                children = self.expand_node(node, initial_prompt, objective)
                new_nodes.extend(children)

            nodes = new_nodes
            if not nodes:
                break

        return best_sequence if best_sequence else []

def evaluate_tot_solution(
    model: Any,
    params: Any,
    tokenizer: Any,
    solution: List[str],
    objective: str,
    key: jax.random.PRNGKey
) -> Tuple[float, str]:
    """
    Evaluate a Tree of Thoughts solution.

    Args:
        model: JAX/Flax model instance.
        params: Model parameters.
        tokenizer: Tokenizer instance.
        solution: Sequence of thoughts.
        objective: Original objective.
        key: JAX random key for generation.

    Returns:
        Tuple of (score, feedback).
    """
    eval_prompt = (
        f"Objective: {objective}\n\n"
        f"Solution steps:\n"
        + "\n".join(f"{i+1}. {s}" for i, s in enumerate(solution))
        + "\n\nProvide:\n1. A score from 0 to 1\n2. Brief feedback"
    )

    input_ids = tokenizer.encode(
        eval_prompt,
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
        temperature=0.3,
        key=key,
        num_return_sequences=1
    )[0]

    response = tokenizer.decode(generated, skip_special_tokens=True)

    try:
        score_str = response.split("Score:")[1].split("\n")[0].strip()
        score = float(score_str)
        score = max(0.0, min(1.0, score))
        feedback = response.split("Feedback:")[1].strip()
    except (IndexError, ValueError):
        score = 0.0
        feedback = "Unable to parse evaluation."

    return score, feedback