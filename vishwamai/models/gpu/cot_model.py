"""
Chain of Thought (CoT) model for VishwamAI, designed to generate reasoning steps before answers.
Inspired by DeepSeek-R1, outputs are structured with <think> and <answer> tags.
Supports deep calculations for tasks like mathematics and coding.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

# Import VishwamAI components
from vishwamai.models.transformer import VishwamAITransformer
from vishwamai.models.gpu.attention import FlashMLAAttention

# Import GPU optimizations
from vishwamai.models.gpu.optimizations.deep_ep.buffer import Buffer
from vishwamai.models.gpu.optimizations.deep_ep.utils import get_num_sms

class CoTModel(nn.Module):
    """
    CoT model extending a transformer to generate reasoning steps and answers.
    """
    def __init__(self, embed_dim=512, num_layers=12, num_heads=8, ff_dim=2048,
                 vocab_size=50000, max_seq_len=512, num_experts=7):
        super(CoTModel, self).__init__()

        self.special_tokens = {
            "think_start": "<think>",
            "think_end": "</think>",
            "answer_start": "<answer>",
            "answer_end": "</answer>"
        }

        attention_kwargs = {"num_experts": num_experts, "taa_kwargs": {"k": 10, "kernel_dim": 256}}
        self.transformer = VishwamAITransformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            max_seq_len=max_seq_len,
            attention_kwargs=attention_kwargs
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_len = max_seq_len
        self.to(self.device)

        self.ep_buffer = None
        self.num_experts = num_experts

    def init_expert_parallel(self, group, hidden_bytes):
        """Initialize Buffer for expert parallelism"""
        if self.ep_buffer is None:
            self.ep_buffer = Buffer(group=group, hidden_bytes=hidden_bytes)
            Buffer.set_num_sms(get_num_sms())

    def expert_parallel_forward(self, x, topk_idx, topk_weights):
        """Forward pass with expert parallelism using DeepEP"""
        num_tokens_per_rank, num_tokens_rdma, num_tokens_expert, is_token_in_rank, event = \
            self.ep_buffer.get_dispatch_layout(
                topk_idx, self.num_experts,
                previous_event=None, async_finish=True
            )

        recv_x, recv_topk_idx, recv_weights, num_recv_expert, handle, event = \
            self.ep_buffer.dispatch(
                x, topk_idx=topk_idx, topk_weights=topk_weights,
                num_tokens_per_rank=num_tokens_per_rank,
                num_tokens_per_rdma_rank=num_tokens_rdma,
                is_token_in_rank=is_token_in_rank,
                num_tokens_per_expert=num_tokens_expert,
                async_finish=True
            )

        return recv_x, handle, event

    def get_expert_assignment(self, x):
        """Placeholder for expert assignment logic"""
        # Simplified: return top-k indices (actual implementation depends on router)
        return torch.topk(torch.rand(x.size(0), self.num_experts, device=self.device), k=2, dim=-1)[1]

    def get_expert_weights(self, x):
        """Placeholder for expert weights"""
        # Simplified: return uniform weights
        return torch.ones(x.size(0), 2, device=self.device) / 2

    def forward(self, input_ids, target_ids=None):
        if target_ids is not None:
            input_to_transformer = torch.cat([input_ids, target_ids[:, :-1]], dim=1)

            if self.ep_buffer is None:
                hidden_bytes = input_to_transformer.size(1) * input_to_transformer.element_size()
                self.init_expert_parallel(torch.distributed.group.WORLD, hidden_bytes)

            exp_output, handle, event = self.expert_parallel_forward(
                input_to_transformer,
                self.get_expert_assignment(input_to_transformer),
                self.get_expert_weights(input_to_transformer)
            )

            logits = self.transformer(exp_output)

            loss = F.cross_entropy(
                logits[:, input_ids.size(1)-1:, :].reshape(-1, self.transformer.vocab_size),
                target_ids[:, 1:].reshape(-1),
                ignore_index=-1
            )
            return loss
        else:
            return self.transformer(input_ids)

    def generate_cot(self, input_text, tokenizer, max_length=512, temperature=0.6, top_p=0.95):
        """Generate CoT output with optimized sampling"""
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        generated_ids = self._sample(input_ids, max_length, temperature, top_p)
        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        return output_text

    def _sample(self, input_ids, max_length, temperature, top_p):
        """Optimized sampling for CoT generation"""
        generated = input_ids
        end_answer_id = self.transformer.tokenizer.encode(self.special_tokens["answer_end"])[0] if \
            hasattr(self.transformer, 'tokenizer') else self.transformer.vocab_size - 1

        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                if generated.size(1) >= max_length:
                    break

                logits = self.transformer(generated)
                next_logits = logits[:, -1, :] / temperature

                # Top-p (nucleus) sampling
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[:, indices_to_remove] = float('-inf')

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

                if next_token.item() == end_answer_id:
                    break

        return generated

def extract_answer(output_text):
    """Extract answer from CoT output"""
    start = output_text.find("<answer>") + len("<answer>")
    end = output_text.find("</answer>")
    if start != -1 and end != -1 and start < end:
        return output_text[start:end].strip()
    return "Answer not found"

def train_cot_model(model, dataloader, optimizer, num_epochs, device):
    """Training loop for CoT model"""
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            optimizer.zero_grad()
            loss = model(input_ids, target_ids)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Example usage
if __name__ == "__main__":
    class MockTokenizer:
        def __init__(self, vocab_size=50000):
            self.vocab_size = vocab_size
            self.special_tokens = {
                "<think>": vocab_size-4, "</think>": vocab_size-3,
                "<answer>": vocab_size-2, "</answer>": vocab_size-1
            }
            self.inverse_vocab = {v: k for k, v in self.special_tokens.items()}
            self.inverse_vocab.update({i: f"token_{i}" for i in range(vocab_size-4)})

        def encode(self, text, return_tensors="pt"):
            tokens = [self.special_tokens.get(text, i) for i in range(5)]
            if return_tensors == "pt":
                return torch.tensor([tokens], dtype=torch.long)
            return tokens

        def decode(self, token_ids, skip_special_tokens=False):
            text = ""
            for token in token_ids:
                if token.item() in self.inverse_vocab:
                    if not skip_special_tokens or token.item() < self.vocab_size-4:
                        text += self.inverse_vocab[token.item()] + " "
            return text.strip()

    tokenizer = MockTokenizer()
    model = CoTModel(vocab_size=tokenizer.vocab_size)

    input_ids = torch.randint(0, tokenizer.vocab_size-4, (10, 20))
    target_ids = torch.cat([
        torch.tensor([[tokenizer.special_tokens["<think>"]]] * 10),
        torch.randint(0, tokenizer.vocab_size-4, (10, 18)),
        torch.tensor([[tokenizer.special_tokens["</think>"]]] * 10),
        torch.tensor([[tokenizer.special_tokens["<answer>"]]] * 10),
        torch.randint(0, tokenizer.vocab_size-4, (10, 5)),
        torch.tensor([[tokenizer.special_tokens["</answer>"]]] * 10)
    ], dim=1)
    dataset = TensorDataset(input_ids, target_ids)
    dataloader = DataLoader(dataset, batch_size=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_cot_model(model, dataloader, optimizer, num_epochs=3, device=model.device)

    input_text = "Solve 2x + 3 = 7"
    output = model.generate_cot(input_text, tokenizer)
    print("Generated CoT:", output)
    print("Extracted Answer:", extract_answer(output))