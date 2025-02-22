"""
Evaluation metrics for Vishwamai model
"""
from typing import Dict, List, Optional, Union
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn.functional as F

def compute_metrics(
    predictions: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    prefix: str = ""
) -> Dict[str, float]:
    """
    Compute classification metrics
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        prefix: Prefix for metric names
        
    Returns:
        Dictionary of metrics
    """
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
        
    # For sequence predictions, take argmax
    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=-1)
        
    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted"
    )
    
    accuracy = accuracy_score(labels, predictions)
    
    metrics = {
        f"{prefix}accuracy": accuracy,
        f"{prefix}precision": precision,
        f"{prefix}recall": recall,
        f"{prefix}f1": f1
    }
    
    return metrics

def calculate_perplexity(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Calculate perplexity from logits and labels
    
    Args:
        logits: Model logits of shape [batch_size, seq_len, vocab_size]
        labels: Label indices of shape [batch_size, seq_len]
        ignore_index: Label index to ignore
        
    Returns:
        Perplexity value
    """
    # Create mask for padding
    mask = (labels != ignore_index).float()
    
    # Calculate cross entropy loss
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=ignore_index,
        reduction="none"
    )
    
    # Reshape loss and apply mask
    loss = loss.view(labels.shape) * mask
    
    # Sum loss and count tokens
    total_loss = loss.sum()
    num_tokens = mask.sum()
    
    # Calculate perplexity
    if num_tokens > 0:
        return torch.exp(total_loss / num_tokens)
    return torch.tensor(float("inf"))

def calculate_sequence_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    Calculate sequence-level accuracy
    
    Args:
        predictions: Predicted token indices
        labels: Target token indices
        ignore_index: Label index to ignore
        
    Returns:
        Sequence accuracy
    """
    mask = (labels != ignore_index)
    correct = (predictions == labels) * mask
    
    # Count sequences where all tokens are correct
    sequence_correct = (correct.sum(dim=1) == mask.sum(dim=1)).float()
    return sequence_correct.mean().item()

def evaluate_generation(
    generated: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Evaluate text generation using various metrics
    
    Args:
        generated: List of generated texts
        references: List of reference texts
        
    Returns:
        Dictionary of metrics
    """
    try:
        from rouge_score import rouge_scorer
        from sacrebleu.metrics import BLEU
    except ImportError:
        raise ImportError(
            "Please install rouge-score and sacrebleu: "
            "pip install rouge-score sacrebleu"
        )
        
    # Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    rouge_scores = [scorer.score(ref, gen) for ref, gen in zip(references, generated)]
    
    rouge_metrics = {
        "rouge1_precision": np.mean([s["rouge1"].precision for s in rouge_scores]),
        "rouge1_recall": np.mean([s["rouge1"].recall for s in rouge_scores]),
        "rouge1_fmeasure": np.mean([s["rouge1"].fmeasure for s in rouge_scores]),
        "rouge2_precision": np.mean([s["rouge2"].precision for s in rouge_scores]),
        "rouge2_recall": np.mean([s["rouge2"].recall for s in rouge_scores]),
        "rouge2_fmeasure": np.mean([s["rouge2"].fmeasure for s in rouge_scores]),
        "rougeL_precision": np.mean([s["rougeL"].precision for s in rouge_scores]),
        "rougeL_recall": np.mean([s["rougeL"].recall for s in rouge_scores]),
        "rougeL_fmeasure": np.mean([s["rougeL"].fmeasure for s in rouge_scores])
    }
    
    # Calculate BLEU score
    bleu = BLEU()
    bleu_score = bleu.corpus_score(generated, [references]).score
    
    metrics = {
        "bleu": bleu_score,
        **rouge_metrics
    }
    
    return metrics

def exact_match_score(
    predictions: List[str],
    references: List[str]
) -> float:
    """
    Calculate exact match score
    
    Args:
        predictions: List of predicted strings
        references: List of reference strings
        
    Returns:
        Exact match score
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) does not match "
            f"number of references ({len(references)})"
        )
        
    matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    return matches / len(predictions)

def token_error_rate(
    predictions: torch.Tensor,
    references: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    Calculate token error rate
    
    Args:
        predictions: Predicted token indices
        references: Reference token indices
        ignore_index: Index to ignore
        
    Returns:
        Token error rate
    """
    mask = (references != ignore_index)
    errors = ((predictions != references) * mask).sum()
    total = mask.sum()
    
    if total == 0:
        return 0.0
        
    return (errors / total).item()
