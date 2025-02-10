import os
import json
from dataclasses import dataclass
from typing import List, Optional, Literal
from argparse import ArgumentParser

import torch
import torch.distributed as dist
from .model import VishwamaiModel, VishwamaiConfig
from .conceptual_tokenizer import ConceptualTokenizer, ConceptualTokenizerConfig

@dataclass
class GenerationConfig:
    max_length: int = 100
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = True
    num_return_sequences: int = 1
    repetition_penalty: float = 1.1
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

class VishwamaiGenerator:
    def __init__(
        self, 
        model: VishwamaiModel,
        tokenizer: ConceptualTokenizer,
        config: Optional[GenerationConfig] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.config.do_sample:
            return logits.argmax(dim=-1)
            
        logits = logits / max(self.config.temperature, 1e-5)
        
        if self.config.top_k > 0:
            v, _ = torch.topk(logits, min(self.config.top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        probs = torch.softmax(logits, dim=-1)
        
        if self.config.top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            probs = probs.masked_fill(indices_to_remove, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def generate(self, prompt: str) -> List[str]:
        """Generate completions for a text prompt."""
        input_ids = self.tokenizer.encode(prompt)
        output_ids = self._generate_tokens(torch.tensor([input_ids], device=self.model.device))
        return [self.tokenizer.decode(ids) for ids in output_ids.tolist()]

    @torch.inference_mode()
    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        max_prompt_len = input_ids.shape[1]
        # Get actual content length by finding first pad token
        prompt_mask = input_ids != self.config.pad_token_id
        actual_lengths = prompt_mask.sum(dim=1)
        max_actual_len = actual_lengths.max().item()
        
        # Calculate output length capped by config limits
        total_len = min(
            self.config.max_length,  # Maximum sequence length from generation config
            self.model.config.max_seq_len  # Maximum length supported by model architecture
        )
        
        # Copy input tokens
        tokens = torch.full(
            (batch_size, total_len),
            self.config.pad_token_id,
            dtype=torch.long,
            device=input_ids.device
        )
        for i in range(batch_size):
            length = actual_lengths[i]
            tokens[i, :length] = input_ids[i, :length]

        # Create or expand attention mask
        attention_mask = torch.zeros(
            (batch_size, total_len),
            dtype=torch.float,
            device=input_ids.device
        )
        for i in range(batch_size):
            length = actual_lengths[i]
            attention_mask[i, :length] = 1.0

        # Track completed sequences and start from actual content length
        finished = torch.tensor([False] * batch_size, device=input_ids.device)
        cur_pos = max_actual_len

        # Generate tokens
        while cur_pos < total_len and not finished.all():
            # Forward pass with attention mask
            # Forward pass
            model_output = self.model(
                tokens[:, :cur_pos],
                attention_mask=attention_mask[:, :cur_pos]
            )
            logits = model_output[:, -1]
            
            # Apply repetition penalty
            if self.config.repetition_penalty != 1.0:
                for i in range(batch_size):
                    scored_tokens = tokens[i, :cur_pos]
                    logits[i, scored_tokens] /= self.config.repetition_penalty
                    
            # Sample next token
            next_token = self.sample(logits)
            tokens[:, cur_pos] = next_token
            attention_mask[:, cur_pos] = 1.0

            # Update finished state
            finished = finished | (next_token == self.config.eos_token_id)
            cur_pos += 1

        # Find sequence end positions
        eos_positions = (tokens == self.config.eos_token_id).float()
        eos_positions[eos_positions.sum(-1) == 0, -1] = 1
        sequence_lengths = eos_positions.argmax(-1)

        # Truncate sequences at EOS
        for b in range(batch_size):
            length = sequence_lengths[b]
            tokens[b, length+1:] = self.config.pad_token_id

        return tokens

def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-length", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    # Initialize model and tokenizer
    model_config = VishwamaiConfig()
    model = VishwamaiModel(model_config)
    tokenizer_config = ConceptualTokenizerConfig.from_pretrained(args.ckpt_path)
    tokenizer = ConceptualTokenizer(tokenizer_config)

    if torch.cuda.is_available():
        model = model.cuda()

    # Load generation config if provided
    if os.path.exists(args.config):
        with open(args.config) as f:
            gen_config = GenerationConfig(**json.load(f))
    else:
        gen_config = GenerationConfig(
            max_length=args.max_length,
            temperature=args.temperature
        )

    generator = VishwamaiGenerator(model, tokenizer, gen_config)

    # Interactive mode
    if args.interactive:
        while True:
            prompt = input(">>> ")
            if prompt.lower() in ["/exit", "quit", "exit"]:
                break

            outputs = generator.generate(prompt)
            for output in outputs:
                print(output)

    # Batch mode
    else:
        with open(args.input_file) as f:
            prompts = [line.strip() for line in f]

        for prompt in prompts:
            outputs = generator.generate(prompt)
            print(f"Prompt: {prompt}")
            for i, output in enumerate(outputs):
                print(f"Output {i+1}: {output}\n")

if __name__ == "__main__":
    main()
