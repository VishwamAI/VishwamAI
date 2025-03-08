# VishwamAI/development/feature_experiments/enhanced_decoder.py
import torch
from torch import nn

class EnhancedDecoder(nn.Module):
    def __init__(self, base_decoder, num_paths=5, threshold=0.7):
        super(EnhancedDecoder, self).__init__()
        self.base_decoder = base_decoder  # Existing decoder module
        self.num_paths = num_paths        # Max number of paths to track
        self.threshold = threshold        # Minimum score to keep a path

    def forward(self, encoder_output, initial_paths, max_seq_len=50):
        paths = initial_paths  # Shape: (num_paths, seq_len)
        path_scores = torch.ones(self.num_paths, device=encoder_output.device)

        for step in range(max_seq_len):
            # Get current sequences as input
            decoder_input = paths[:, :step+1]
            # Decode next token for all paths
            logits = self.base_decoder(decoder_input, encoder_output)[:, -1, :]
            probs = torch.softmax(logits, dim=-1)

            # Expand paths with top-k tokens
            top_probs, top_indices = torch.topk(probs, k=self.num_paths, dim=-1)
            new_paths = []
            new_scores = []

            for i in range(self.num_paths):
                for j in range(self.num_paths):
                    new_path = torch.cat([paths[i], top_indices[i, j].unsqueeze(0)], dim=0)
                    new_score = path_scores[i] * top_probs[i, j]
                    if new_score > self.threshold:
                        new_paths.append(new_path)
                        new_scores.append(new_score)

            # Keep top num_paths
            if len(new_paths) > self.num_paths:
                top_indices = torch.topk(torch.tensor(new_scores), k=self.num_paths).indices
                paths = torch.stack([new_paths[idx] for idx in top_indices])
                path_scores = torch.tensor([new_scores[idx] for idx in top_indices])
            else:
                paths = torch.stack(new_paths)
                path_scores = torch.tensor(new_scores)

        return paths, path_scores

# Usage: enhanced_decoder = EnhancedDecoder(base_decoder_module, num_paths=5)