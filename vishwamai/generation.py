import torch
import torch.nn.functional as F
import logging

class BeamSearchGenerator:
    def __init__(self, model, tokenizer, beam_size=4, length_penalty=1.0):
        """
        Initialize the BeamSearchGenerator.
        
        Args:
            model: The language model.
            tokenizer: Tokenizer instance.
            beam_size (int, optional): Number of beams. Defaults to 4.
            length_penalty (float, optional): Length penalty. Defaults to 1.0.
        """
        try:
            self.model = model
            self.tokenizer = tokenizer
            self.beam_size = beam_size
            self.length_penalty = length_penalty
        except Exception as e:
            logging.error(f"Error initializing BeamSearchGenerator: {e}")
            raise

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        min_length: int = 0,
        num_return_sequences: int = 1
    ) -> torch.Tensor:
        """
        Generate sequences using beam search.
        
        Args:
            input_ids (torch.Tensor): Initial input tokens.
            max_length (int): Maximum sequence length.
            min_length (int, optional): Minimum sequence length. Defaults to 0.
            num_return_sequences (int, optional): Number of sequences to return. Defaults to 1.
        
        Returns:
            torch.Tensor: Generated token IDs.
        """
        try:
            batch_size = input_ids.size(0)
            device = input_ids.device
            vocab_size = self.model.config.vocab_size

            # Initialize beams
            beams = [([token], 0.0) for token in input_ids[0]]
            
            for _ in range(max_length):
                all_candidates = []
                for beam, score in beams:
                    if beam[-1] == self.tokenizer.eos_token_id:
                        all_candidates.append((beam, score))
                        continue
                    logits = self.model(torch.tensor([beam]).to(device))
                    log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                    topk_log_probs, topk_indices = torch.topk(log_probs, self.beam_size)
                    for log_prob, idx in zip(topk_log_probs[0], topk_indices[0]):
                        candidate = (beam + [idx.item()], score + log_prob.item())
                        all_candidates.append(candidate)
                # Select top beams
                ordered = sorted(all_candidates, key=lambda tup: tup[1]/(len(tup[0])**self.length_penalty), reverse=True)
                beams = ordered[:self.beam_size]
                # Check if all beams have ended
                if all([beam[-1] == self.tokenizer.eos_token_id for beam, _ in beams]):
                    break
                if len(beams) == 1 and beams[0][0][-1] == self.tokenizer.eos_token_id:
                    break
            # Select the best beam
            best_beam = sorted(beams, key=lambda tup: tup[1], reverse=True)[0][0]
            return torch.tensor(best_beam).unsqueeze(0).to(device)
        except Exception as e:
            logging.error(f"Error during beam search generation: {e}")
            raise
