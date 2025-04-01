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
            
    def add_child(self, child: 'ThoughtNode') -> None:
        """Add a child node."""
        self.children.append(child)
        child.parent = self
        child.depth = self.depth + 1
        
    def get_path(self) -> List[str]:
        """Get path from root to this node."""
        path = []
        current = self
        while current:
            path.insert(0, current.thought)
            current = current.parent
        return path

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
        """Initialize TreeOfThoughts."""
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
        """Generate multiple potential thought branches."""
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
                max_length=len(input_ids) + 100,  # Generate reasonable length continuation
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
        """Estimate value of a thought sequence."""
        # Prepare evaluation prompt
        eval_prompt = (
            f"Objective: {objective}\n\n"
            f"Reasoning steps:\n" + "\n".join(thought_sequence) + "\n\n"
            "Rate how well this reasoning achieves the objective from 0.0 to 1.0:"
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
            temperature=0.1,  # Lower temperature for more focused rating
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
        """Expand a node by generating child thoughts."""
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
            thought_sequence = node.get_path() + [thought]
            value = self.estimate_value(thought_sequence, objective)
            child = ThoughtNode(
                thought=thought,
                value=value,
                parent=node,
                depth=node.depth + 1
            )
            children.append(child)

        # Sort children by value
        children.sort(key=lambda x: x.value, reverse=True)
        
        # Keep only top beam_width children
        children = children[:self.beam_width]
        node.children = children
        
        return children

    def search(
        self,
        initial_prompt: str,
        objective: str,
        max_steps: int = 10
    ) -> List[str]:
        """Perform tree search to find solution path."""
        # Generate initial thoughts
        initial_thoughts = self.generate_thoughts(
            initial_prompt,
            num_thoughts=self.max_branches
        )
        
        # Create root nodes
        nodes = []
        for thought in initial_thoughts:
            value = self.estimate_value([thought], objective)
            node = ThoughtNode(thought=thought, value=value)
            nodes.append(node)
            
        # Sort and prune to beam width
        nodes.sort(key=lambda x: x.value, reverse=True)
        nodes = nodes[:self.beam_width]
        
        best_sequence = []
        best_value = 0.0
        
        # Perform beam search
        for _ in range(max_steps):
            if not nodes:
                break
                
            # Track best sequence seen so far
            if nodes[0].value > best_value:
                best_value = nodes[0].value
                best_sequence = nodes[0].get_path()

            new_nodes = []
            for node in nodes:
                children = self.expand_node(node, initial_prompt, objective)
                new_nodes.extend(children)
                
            if not new_nodes:
                break
                
            # Sort all new nodes by value
            new_nodes.sort(key=lambda x: x.value, reverse=True)
            
            # Keep top beam_width nodes
            nodes = new_nodes[:self.beam_width]
            
        return best_sequence if best_sequence else []

def evaluate_tot_solution(
    model: Any,
    params: Any,
    tokenizer: Any,
    solution: List[str],
    objective: str,
    key: jax.random.PRNGKey
) -> Tuple[float, str]:
    """Evaluate a ToT solution."""
    # Prepare evaluation prompt
    eval_prompt = (
        f"Objective: {objective}\n\n"
        f"Solution:\n" + "\n".join(solution) + "\n\n"
        "Evaluate this solution. First give a score from 0.0 to 1.0, "
        "then explain your rating:"
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
        temperature=0.7,
        key=key,
        num_return_sequences=1
    )[0]

    response = tokenizer.decode(generated, skip_special_tokens=True)
    
    try:
        score = float(next(
            float(s) for s in response.split() if s.replace(".", "").isdigit()
        ))
        score = max(0.0, min(1.0, score))
    except (StopIteration, ValueError):
        score = 0.0
        
    return score, response