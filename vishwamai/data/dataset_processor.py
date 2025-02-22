"""Dataset processors for different benchmarks"""
from typing import Dict, List, Optional
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer

class BenchmarkProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def process_gsm8k(self, split: str = "train") -> Dataset:
        """Process GSM8K dataset"""
        dataset = load_dataset("gsm8k", "main")[split]
        
        def format_gsm8k(example):
            return {
                "input_text": f"Solve this math problem step by step:\n{example['question']}",
                "target_text": example['answer'].split("####")[1].strip(),
                "solution": example['answer'].split("####")[0].strip()
            }
        
        return dataset.map(format_gsm8k)

    def process_mmlu(self, subjects: Optional[List[str]] = None) -> Dataset:
        """Process MMLU dataset"""
        if subjects is None:
            subjects = ["mathematics", "computer_science", "physics"]
        
        datasets = []
        for subject in subjects:
            dataset = load_dataset("cais/mmlu", subject)
            datasets.append(dataset)
        
        def format_mmlu(example):
            options = ["A", "B", "C", "D"]
            formatted_options = "\n".join(
                f"{opt}) {example[opt]}" for opt in options
            )
            return {
                "input_text": f"Question: {example['question']}\n\nOptions:\n{formatted_options}",
                "target_text": options[example['answer']],
                "subject": example['subject']
            }
        
        combined = datasets[0]
        for ds in datasets[1:]:
            combined = combined.concatenate(ds)
        
        return combined.map(format_mmlu)

    def process_mmmu(self) -> Dataset:
        """Process MMMU dataset"""
        dataset = load_dataset("MMMU/MMMU")
        
        def format_mmmu(example):
            return {
                "input_text": f"Problem: {example['problem']}\nContext: {example['context']}",
                "target_text": example['solution'],
                "domain": example['domain']
            }
        
        return dataset.map(format_mmmu)

    def tokenize_function(self, examples):
        """Tokenize inputs and targets"""
        model_inputs = self.tokenizer(
            examples["input_text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples["target_text"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs