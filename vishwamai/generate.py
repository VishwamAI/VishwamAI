import os
import json
from dataclasses import dataclass
from typing import List, Optional, Literal
from argparse import ArgumentParser

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from .model import VishwamaiModel, VishwamaiConfig

@dataclass
class GenerationConfig:
    max_new_tokens: int = 200
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

class VishwamaiGenerator:
    def __init__(
        self, 
        model: VishwamaiModel, 
        config: Optional[GenerationConfig] = None
    ):
        self.model = model
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

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_new_tokens: Optional[int] = None,
    ) -> List[List[int]]:
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
            
        batch_size = len(prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        total_len = min(self.model.config.max_seq_len, max_new_tokens + max_prompt_len)
        
        # Initialize token buffer
        tokens = torch.full(
            (batch_size, total_len), 
            self.config.pad_token_id, 
            dtype=torch.long, 
            device=self.model.device
        )
        
        # Copy prompt tokens
        for i, t in enumerate(prompt_tokens):
            tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=self.model.device)
            
        # Track completed sequences
        finished = torch.tensor([False] * batch_size, device=self.model.device)
        prompt_mask = tokens != self.config.pad_token_id
        
        # Generate tokens
        prev_pos = 0
        for cur_pos in range(max_prompt_len, total_len):
            logits = self.model(tokens[:, :cur_pos])[:, -1]
            
            # Apply repetition penalty
            if self.config.repetition_penalty != 1.0:
                for i in range(batch_size):
                    scored_tokens = tokens[i, :cur_pos]
                    logits[i, scored_tokens] /= self.config.repetition_penalty
                    
            next_token = self.sample(logits)
            next_token = torch.where(
                prompt_mask[:, cur_pos], 
                tokens[:, cur_pos], 
                next_token
            )
            tokens[:, cur_pos] = next_token
            
            # Check for completed sequences
            finished |= (~prompt_mask[:, cur_pos] & (next_token == self.config.eos_token_id))
            if finished.all():
                break
                
        # Extract completions
        completions = []
        for i, toks in enumerate(tokens.tolist()):
            completion = toks[len(prompt_tokens[i]):cur_pos + 1]
            if self.config.eos_token_id in completion:
                completion = completion[:completion.index(self.config.eos_token_id)]
            completions.append(completion)
            
        return completions

def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    
    # Initialize distributed setup if needed
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(rank)
        
    # Load model and tokenizer
    model_config = VishwamaiConfig()
    model = VishwamaiModel(model_config).to(rank)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    
    if os.path.exists(args.config):
        with open(args.config) as f:
            gen_config = GenerationConfig(**json.load(f))
    else:
        gen_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        
    generator = VishwamaiGenerator(model, gen_config)

    # Interactive mode
    if args.interactive:
        while True:
            prompt = input(">>> ")
            if prompt.lower() in ["/exit", "quit", "exit"]:
                break
                
            tokens = tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
            completion = generator.generate([tokens])[0]
            print(tokenizer.decode(completion))
            
    # Batch mode
    else:
        with open(args.input_file) as f:
            prompts = [line.strip() for line in f]
            
        tokens = [tokenizer.encode(p) for p in prompts]
        completions = generator.generate(tokens)
        
        for prompt, completion in zip(prompts, completions):
            print(f"Prompt: {prompt}")
            print(f"Completion: {tokenizer.decode(completion)}\n")
            
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()