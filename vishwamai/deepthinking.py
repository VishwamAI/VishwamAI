# deepthinking.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import json
import re
from collections import defaultdict
from .model import VishwamaiModel
from .conceptual_tokenizer import ConceptualTokenizer
from .architecture import init_model  # Import init_model function

THINK_START = "<think>"
THINK_END = "</think>"
ANSWER_START = "<answer>"
ANSWER_END = "</answer>"
REFLECT_START = "<reflect>"
REFLECT_END = "</reflect>"
STEP_START = "<step>"
STEP_END = "</step>"

class CoTGenerationWrapper(nn.Module):
    """Wrapper for VishwamaiModel to handle Chain-of-Thought generation with self-reflection"""
    def __init__(self, model: VishwamaiModel, tokenizer: ConceptualTokenizer, num_self_reflect_steps: int = 2):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = tokenizer.config.max_length
        self.device = model.config.torch_device
        self.model.to(self.device)
        
        # Add special tokens for CoT formatting with proper fallback
        def get_valid_token_id(name):
            token_id = tokenizer.concept_embeddings.get(name)
            # Return unk_id for invalid token IDs (None or negative)
            if token_id is None or token_id < 0 or token_id >= tokenizer.config.vocab_size:
                return tokenizer.unk_id
            return token_id

        self.think_start_id = get_valid_token_id("[CONCEPT_THINK_START]")
        self.think_end_id = get_valid_token_id("[CONCEPT_THINK_END]")
        self.answer_start_id = get_valid_token_id("[CONCEPT_ANSWER_START]")
        self.answer_end_id = get_valid_token_id("[CONCEPT_ANSWER_END]")
        self.reflect_start_id = get_valid_token_id("[CONCEPT_REFLECT_START]")
        self.reflect_end_id = get_valid_token_id("[CONCEPT_REFLECT_END]")
        self.step_start_id = get_valid_token_id("[CONCEPT_STEP_START]")
        self.step_end_id = get_valid_token_id("[CONCEPT_STEP_END]")
        self.num_self_reflect_steps = num_self_reflect_steps

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_samples: int = 1
    ) -> List[Dict[str, str]]:
        """Generate CoT responses with structured formatting"""
        prompt = f"User: {prompt}\nAssistant:"
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        
        all_sequences = []
        for _ in range(num_samples):
            current_ids = torch.tensor([input_ids], device=self.device)
            finished = False
            think_buffer = []
            answer_buffer = []
            in_think = False
            in_answer = False
            in_reflect = False
            in_step = False
            current_step = 1
            reflection_count = 0
            steps = []

            for _ in range(max_new_tokens):
                with torch.no_grad():
                    logits = self.model(current_ids)[0, -1]

                # Apply temperature and top-p filtering
                logits = logits / temperature
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = -float('inf')

                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Update state machines for CoT formatting
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=-1)
                decoded_token = self.tokenizer.decode([next_token.item()])

                # Handle special tokens and step tracking
                if decoded_token == THINK_START:
                    in_think = True
                    think_buffer = []
                elif decoded_token == THINK_END:
                    in_think = False
                    # After thinking, potentially trigger reflection
                    if reflection_count < self.num_self_reflect_steps:
                        current_ids = torch.cat([
                            current_ids,
                            torch.tensor([[self.reflect_start_id]], device=self.device)
                        ], dim=-1)
                        in_reflect = True
                        reflection_count += 1
                elif decoded_token == ANSWER_START:
                    in_answer = True
                    answer_buffer = []
                elif decoded_token == ANSWER_END:
                    in_answer = False
                    finished = True
                elif decoded_token == REFLECT_START:
                    in_reflect = True
                elif decoded_token == REFLECT_END:
                    in_reflect = False
                    # After reflection, continue with next step
                    if in_think:
                        current_ids = torch.cat([
                            current_ids,
                            torch.tensor([[self.step_start_id]], device=self.device)
                        ], dim=-1)
                elif decoded_token == STEP_START:
                    in_step = True
                    steps.append(f"Step {current_step}: ")
                elif decoded_token == STEP_END:
                    in_step = False
                    current_step += 1
                else:
                    if in_think:
                        think_buffer.append(decoded_token)
                    elif in_answer:
                        answer_buffer.append(decoded_token)
                    elif in_reflect:
                        # Store reflection in think buffer
                        think_buffer.append(f"[Reflection {reflection_count}] {decoded_token}")
                    elif in_step:
                        steps[-1] += decoded_token

                if finished or current_ids.size(-1) >= self.max_length:
                    break

            # Post-process output with steps and reflections
            full_output = self.tokenizer.decode(current_ids[0].tolist())
            thought = "".join(think_buffer).strip()
            answer = "".join(answer_buffer).strip()
            
            # Extract reflections
            reflections = []
            for line in thought.split('\n'):
                if line.startswith('[Reflection'):
                    reflections.append(line)
            
            all_sequences.append({
                "full_output": full_output,
                "thought": thought,
                "answer": answer,
                "steps": steps,
                "reflections": reflections
            })
            
        return all_sequences

class GRPOTrainer:
    """Implementation of Group Relative Policy Optimization"""
    def __init__(
        self,
        model: VishwamaiModel,
        tokenizer: ConceptualTokenizer,
        reward_fns: Dict[str, callable],
        gamma: float = 0.99,
        beta: float = 0.1,
        eps_clip: float = 0.2,
        group_size: int = 4
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fns = reward_fns
        self.gamma = gamma
        self.beta = beta
        self.eps_clip = eps_clip
        self.group_size = group_size
        self.device = model.config.torch_device
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
        
        # Initialize reference model for KL divergence
        self.ref_model = VishwamaiModel(model.config)
        self.ref_model.to(self.device)  # Use config device
        self.ref_model.load_state_dict(model.state_dict())
        self.ref_model.eval()

    def compute_rewards(self, responses: List[str]) -> Dict[str, torch.Tensor]:
        """Compute rewards for a batch of responses"""
        rewards = defaultdict(list)
        for response in responses:
            for name, fn in self.reward_fns.items():
                rewards[name].append(fn(response))
        
        # Normalize rewards with proper handling of constant values
        normalized = {}
        for name, values in rewards.items():
            tensor = torch.tensor(values, device=self.device)
            mean = tensor.mean()
            std = tensor.std()
            
            if std == 0:  # Handle constant rewards
                # For constant rewards, generate unit variance noise with zero mean
                noise = torch.randn_like(tensor)
                noise = (noise - noise.mean()) / (noise.std() + 1e-8)  # Normalize to zero mean and unit variance
                normalized[name] = noise
            else:
                # Standard normalization (zero mean, unit variance)
                normalized[name] = (tensor - mean) / (std + 1e-8)  # Add epsilon to avoid division by zero
            
        return normalized

    def train_step(self, prompts: List[str]):
        # Generate responses
        self.model.train()
        all_logprobs = []
        all_values = []
        all_rewards = []
        
        # Process in groups
        for i in range(0, len(prompts), self.group_size):
            group_prompts = prompts[i:i+self.group_size]
            
            # Generate multiple responses per prompt
            with torch.no_grad():
                group_responses = []
                for prompt in group_prompts:
                    outputs = self.model.generate(prompt, num_samples=self.group_size)
                    group_responses.extend([r["full_output"] for r in outputs])
            
            # Compute rewards
            rewards = self.compute_rewards(group_responses)
            total_rewards = sum(rewards.values())  # Sum normalized rewards
            
            # Get reference model logprobs
            with torch.no_grad():
                ref_logprobs = self._get_logprobs(group_responses, self.ref_model)
                
            # Get current model logprobs
            current_logprobs = self._get_logprobs(group_responses, self.model)
            
            # Compute advantages
            total_rewards = torch.tensor(total_rewards, device=self.device)
            advantages = (total_rewards - total_rewards.mean()) / (total_rewards.std() + 1e-8)
            
            # Compute policy loss
            ratios = torch.exp(current_logprobs - ref_logprobs)
            clipped_ratios = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip)
            policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()
            
            # Compute KL penalty
            kl_penalty = (current_logprobs - ref_logprobs).mean() * self.beta
            
            # Total loss
            loss = policy_loss + kl_penalty
            
            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return loss.item()

    def _get_logprobs(self, responses: List[str], model: VishwamaiModel) -> torch.Tensor:
        """Compute log probabilities for given responses"""
        batch = self.tokenizer.batch_encode_with_concepts(responses)
        input_ids = batch['input_ids'].to(model.device)
        
        with torch.no_grad() if model is self.ref_model else torch.enable_grad():
            logits = model(input_ids)
            logprobs = torch.log_softmax(logits, dim=-1)
            
        # Gather logprobs for actual tokens
        selected_logprobs = logprobs[:, :-1].gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze()
        return selected_logprobs.mean(dim=-1)

class ReasoningDataset(Dataset):
    """Dataset for storing reasoning problems and solutions"""
    def __init__(self, problems: List[Dict], tokenizer: ConceptualTokenizer):
        self.tokenizer = tokenizer
        self.data = []
        
        for problem in problems:
            formatted = f"User: {problem['question']}\nAssistant: "
            formatted += f"{THINK_START}{problem['thought']}{THINK_END}"
            formatted += f"{ANSWER_START}{problem['answer']}{ANSWER_END}"
            
            encoded = tokenizer.encode(formatted)
            self.data.append({
                "input_ids": encoded,
                "concept_ids": problem.get("concepts", [])
            })
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def create_format_reward_fn(tokenizer: ConceptualTokenizer) -> callable:
    """Create reward function for format compliance with step tracking and reflection"""
    def format_reward(response: str) -> float:
        has_think = THINK_START in response and THINK_END in response
        has_answer = ANSWER_START in response and ANSWER_END in response
        has_reflection = REFLECT_START in response and REFLECT_END in response
        has_steps = STEP_START in response and STEP_END in response
        
        order_correct = all([
            response.find(THINK_START) < response.find(ANSWER_START),
            response.find(STEP_START) > response.find(THINK_START),
            response.find(REFLECT_START) > response.find(THINK_START)
        ])
        
        # Count number of steps
        step_count = response.count(STEP_START)
        step_bonus = min(step_count / 3.0, 1.0)  # Reward up to 3 steps
        
        # Base score
        base_score = float(has_think and has_answer and order_correct)
        
        # Additional rewards for reflection and steps
        reflection_bonus = 0.2 if has_reflection else 0.0
        step_organization_bonus = 0.2 if has_steps else 0.0
        
        return base_score + reflection_bonus + step_organization_bonus + step_bonus * 0.2
    return format_reward

def create_accuracy_reward_fn(answers: Dict[str, str]) -> callable:
    """Create reward function for answer correctness"""
    def accuracy_reward(response: str) -> float:
        answer = response.split(ANSWER_END)[0].split(ANSWER_START)[-1].strip()
        return float(answer == answers.get(hash(response), ""))
    return accuracy_reward

def distill_r1(
    teacher: VishwamaiModel,
    student: VishwamaiModel,
    dataset: ReasoningDataset,
    batch_size: int = 4,
    num_epochs: int = 3
):
    """Distill knowledge from DeepSeek-R1 to smaller model"""
    teacher.eval()
    student.train()
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5)
    loss_fn = nn.KLDivLoss(reduction="batchmean")
    
    for epoch in range(num_epochs):
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i:i+batch_size]
            input_ids = torch.stack([torch.tensor(d["input_ids"]) for d in batch]).to(teacher.config.torch_device)
            
            # Get teacher outputs
            with torch.no_grad():
                teacher_logits = teacher(input_ids)
                
            # Get student outputs
            student_logits = student(input_ids)
            
            # Compute distillation loss
            loss = loss_fn(
                torch.log_softmax(student_logits, dim=-1),
                torch.softmax(teacher_logits, dim=-1)
            )
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

# Example usage:
# tokenizer = ConceptualTokenizer.from_pretrained("path/to/tokenizer")
# model = init_model("config_23b")
# trainer = GRPOTrainer(model, tokenizer, {...})
# dataset = ReasoningDataset(problems, tokenizer)
# distill_r1(teacher_model, student_model, dataset)
