# /home/kasinadhsarma/VishwamAI/vishwamai/inference/inference_cot.py
"""
Inference interface for the Chain of Thought (CoT) model in VishwamAI.
Supports deep and normal thinking modes for generating thought sequences.
"""

import torch
import torch_xla.core.xla_model as xm
from typing import List, Tuple
from .optimized_inference import OptimizedInference

class CoTInference:
    """
    Interface for running inference on the CoT model.
    Provides methods for generating thought sequences with deep or normal thinking.
    """
    def __init__(self, model, device='tpu', precision='bf16'):
        self.model = model
        self.optimizer = OptimizedInference()
        self.optimizer.set_device(device)
        self.optimizer.set_precision(precision)
        self.optimizer.optimize_model(self.model)
        self.device = self.optimizer.device
        self.thought_sequence = []

    def generate_thoughts(self, input_data: str, thinking_mode: str = 'normal') -> Tuple[List[str], str]:
        """
        Generate a sequence of thoughts leading to the final answer.

        Args:
            input_data (str): The input query or problem.
            thinking_mode (str): 'normal' for standard reasoning, 'deep' for detailed reasoning.

        Returns:
            Tuple[List[str], str]: Thought sequence and final answer.
        """
        self.thought_sequence = []
        processed_input = self._preprocess_input(input_data)
        current_thought = processed_input
        max_steps = 5 if thinking_mode == 'normal' else 10  # Deep mode allows more steps
        temperature = 0.7 if thinking_mode == 'normal' else 1.0  # Deep mode explores more

        for _ in range(max_steps):
            if self._is_final_thought(current_thought):
                break
            next_thought = self._next_thought(current_thought, temperature)
            self.thought_sequence.append(current_thought)
            current_thought = next_thought

        final_answer = self._get_answer(current_thought)
        self.thought_sequence.append(current_thought)
        return self.thought_sequence, final_answer

    def _preprocess_input(self, input_data: str) -> str:
        """Preprocess the input data for the CoT model."""
        return f"Thinking about: {input_data}"

    def _next_thought(self, current_thought: str, temperature: float) -> str:
        """Generate the next thought using the model."""
        input_tensor = torch.tensor([ord(c) for c in current_thought], device=self.device).unsqueeze(0)
        with torch.no_grad():
            output = self.optimizer.run_model(self.model, input_tensor, temperature=temperature)
        return f"{current_thought} -> {self._decode_output(output)}"

    def _is_final_thought(self, thought: str) -> bool:
        """Check if the current thought is the final one."""
        return "answer" in thought.lower() or "final" in thought.lower()

    def _get_answer(self, final_thought: str) -> str:
        """Extract the final answer from the thought."""
        return final_thought.split(" -> ")[-1]

    def get_thought_sequence(self) -> List[str]:
        """Return the current thought sequence."""
        return self.thought_sequence

if __name__ == "__main__":
    # Dummy model for testing
    class DummyCoTModel(torch.nn.Module):
        def forward(self, x, temperature=1.0):
            return torch.tensor([ord('n'), ord('e'), ord('x'), ord('t')], device=x.device)

        def _decode_output(self, output):
            return ''.join(chr(int(x)) for x in output)

    model = DummyCoTModel()
    cot_inf = CoTInference(model)
    input_data = "What is the capital of France?"
    thoughts, answer = cot_inf.generate_thoughts(input_data, thinking_mode='deep')
    print("Thought Sequence:", thoughts)
    print("Final Answer:", answer)