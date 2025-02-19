import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
from datasets import load_dataset
from tqdm import tqdm

def evaluate_model(
    model: torch.nn.Module,
    datasets: List[Tuple[str, str]],
    batch_size: int = 16,
    device: torch.device = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on multiple benchmark datasets.
    
    Args:
        model: Model to evaluate
        datasets: List of (dataset_name, split) tuples
        batch_size: Batch size for evaluation
        device: Device to use for evaluation
        
    Returns:
        Dictionary of evaluation metrics for each dataset
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model.eval()
    results = {}
    
    for dataset_name, split in datasets:
        try:
            # Load dataset
            dataset = load_dataset(dataset_name, split=split)
            if isinstance(dataset, dict):
                dataset = dataset[split]
                
            # Setup dataloader
            from torch.utils.data import DataLoader
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            # Initialize metrics
            metrics = {
                'accuracy': 0.0,
                'perplexity': 0.0,
                'calibration_error': 0.0,
                'reasoning_score': 0.0
            }
            
            total_samples = 0
            
            # Evaluate
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    
                    if 'labels' in batch:
                        # Calculate accuracy
                        predictions = outputs['logits'].argmax(dim=-1)
                        correct = (predictions == batch['labels']).float().sum()
                        metrics['accuracy'] += correct.item()
                        
                        # Calculate perplexity
                        loss = F.cross_entropy(outputs['logits'], batch['labels'])
                        metrics['perplexity'] += torch.exp(loss).item() * len(batch['labels'])
                        
                        # Calculate calibration error
                        probs = F.softmax(outputs['logits'], dim=-1)
                        conf = probs.max(dim=-1)[0]
                        metrics['calibration_error'] += torch.abs(conf - (predictions == batch['labels']).float()).sum().item()
                        
                        # Calculate reasoning score if available
                        if 'reasoning_outputs' in outputs:
                            metrics['reasoning_score'] += outputs['reasoning_outputs']['score'].mean().item() * len(batch['labels'])
                            
                    total_samples += len(batch['input_ids'])
                    
            # Average metrics
            if total_samples > 0:
                metrics['accuracy'] /= total_samples
                metrics['perplexity'] /= total_samples
                metrics['calibration_error'] /= total_samples
                metrics['reasoning_score'] /= total_samples
                
            results[dataset_name] = metrics
            
        except Exception as e:
            print(f"Error evaluating {dataset_name}: {e}")
            continue
            
    return results
