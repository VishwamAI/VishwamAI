import torch
from transformers import AutoTokenizer
from ..model import VishwamAIModel
from ..neural_memory import ReasoningMemoryTransformer, MemoryConfig
from ..tree_of_thoughts import TreeOfThoughts, TreeConfig
from ..cache_augmentation import DifferentiableCacheAugmentation, CacheConfig
from typing import Optional, Dict, List

class EnhancedVishwamAI:
    """Enhanced VishwamAI model with memory, tree search, and cache components."""
    
    def __init__(self, 
                 model_path: str,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 use_memory: bool = True,
                 use_tree: bool = True,
                 use_cache: bool = True):
        
        # Load base model
        self.model = VishwamAIModel.from_pretrained(model_path)
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = device
        
        # Initialize enhancement components
        if use_memory:
            self.memory = ReasoningMemoryTransformer.from_pretrained(model_path)
            self.memory.to(device)
        else:
            self.memory = None
            
        if use_tree:
            self.tree = TreeOfThoughts.from_pretrained(model_path, model=self.model)
            self.tree.to(device)
        else:
            self.tree = None
            
        if use_cache:
            self.cache = DifferentiableCacheAugmentation.from_pretrained(model_path)
            self.cache.to(device)
        else:
            self.cache = None
            
    def generate_response(self, 
                         prompt: str,
                         max_length: int = 2048,
                         temperature: float = 0.7,
                         num_beams: int = 4,
                         **kwargs) -> str:
        """Generate enhanced response using all components."""
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate with enhancements
        with torch.inference_mode():
            # Get base model hidden states
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            
            # Apply memory enhancement
            if self.memory is not None:
                hidden_states = self.memory(hidden_states)
                
            # Apply tree of thoughts
            if self.tree is not None:
                hidden_states = self.tree(hidden_states)
                
            # Apply cache augmentation
            if self.cache is not None:
                hidden_states = self.cache(hidden_states)
            
            # Generate final output
            enhanced_outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                temperature=temperature,
                num_beams=num_beams,
                past_key_values=outputs.past_key_values,
                hidden_states=hidden_states,
                **kwargs
            )
            
        response = self.tokenizer.decode(enhanced_outputs[0], skip_special_tokens=True)
        return response
    
    def batch_generate(self, 
                      prompts: List[str],
                      **kwargs) -> List[str]:
        """Generate responses for multiple prompts."""
        return [self.generate_response(prompt, **kwargs) for prompt in prompts]

def example_usage():
    """Example usage of the enhanced model."""
    
    # Initialize model
    model = EnhancedVishwamAI(
        model_path="kasinadhsarma/vishwamai-model",
        use_memory=True,
        use_tree=True,
        use_cache=True
    )
    
    # Single response generation
    prompt = """
    Solve this complex mathematical problem step by step:
    A company's profit increased by 20% in the first year and decreased by 10% in the second year.
    If the final profit is $108,000, what was the initial profit?
    """
    
    response = model.generate_response(
        prompt,
        max_length=512,
        temperature=0.7,
        num_beams=4
    )
    print(f"Problem:\n{prompt}\n\nSolution:\n{response}\n")
    
    # Batch generation
    prompts = [
        "Explain quantum entanglement to a high school student.",
        "Write a recursive Python function to calculate the nth Fibonacci number.",
        "Describe three key differences between supervised and unsupervised learning."
    ]
    
    responses = model.batch_generate(
        prompts,
        max_length=256,
        temperature=0.8
    )
    
    for prompt, response in zip(prompts, responses):
        print(f"Prompt: {prompt}\nResponse: {response}\n")

if __name__ == "__main__":
    example_usage()
