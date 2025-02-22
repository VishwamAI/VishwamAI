"""Validation manager for benchmark datasets"""
import torch
from typing import Dict, List
import numpy as np
from datasets import Dataset
import wandb

class BenchmarkValidator:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def validate_gsm8k(self, dataset: Dataset) -> Dict[str, float]:
        """Validate on GSM8K benchmark"""
        correct = 0
        total = 0
        
        self.model.eval()
        with torch.no_grad():
            for example in dataset:
                input_ids = self.tokenizer(
                    example["input_text"],
                    return_tensors="pt",
                    truncation=True
                ).input_ids.to(self.device)
                
                outputs = self.model.generate(
                    input_ids,
                    max_length=200,
                    num_beams=4,
                    early_stopping=True
                )
                
                predicted = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                correct += self.check_gsm8k_answer(predicted, example["target_text"])
                total += 1
        
        accuracy = correct / total
        return {
            "gsm8k_accuracy": accuracy,
            "gsm8k_correct": correct,
            "gsm8k_total": total
        }
    
    def validate_mmlu(self, dataset: Dataset) -> Dict[str, float]:
        """Validate on MMLU benchmark"""
        results = {}
        subjects = dataset.unique("subject")
        
        for subject in subjects:
            subject_data = dataset.filter(lambda x: x["subject"] == subject)
            correct = 0
            total = 0
            
            for example in subject_data:
                input_ids = self.tokenizer(
                    example["input_text"],
                    return_tensors="pt",
                    truncation=True
                ).input_ids.to(self.device)
                
                outputs = self.model.generate(
                    input_ids,
                    max_length=10,
                    num_beams=1
                )
                
                predicted = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                correct += predicted.strip() == example["target_text"].strip()
                total += 1
            
            results[f"mmlu_{subject}_accuracy"] = correct / total
        
        results["mmlu_average"] = np.mean(list(results.values()))
        return results
    
    @staticmethod
    def check_gsm8k_answer(predicted: str, target: str) -> bool:
        """Check if GSM8K answer is correct"""
        try:
            pred_num = float(''.join(filter(str.isdigit, predicted)))
            target_num = float(''.join(filter(str.isdigit, target)))
            return abs(pred_num - target_num) < 1e-6
        except:
            return False