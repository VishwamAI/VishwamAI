"""
Example usage of VishwamAI model demonstrating various features and capabilities.
"""

import torch
from vishwamai import VishwamAI, ModelConfig
from vishwamai.extensions.ethical_framework import EthicalConfig
from vishwamai.extensions.tree_of_thoughts import TreeConfig, RewardConfig

def main():
    # Basic model configuration
    config = ModelConfig(
        vocab_size=50257,
        hidden_size=2048,
        num_layers=24,
        num_heads=16,
        use_mla=True,  # Enable Multi-Level Attention
        use_memory=True,  # Enable Neural Memory
        memory_size=1024,
        use_moe=True,  # Enable Mixture of Experts
        num_experts=8,
        expert_capacity=128,
        use_ethical_framework=True,
        enable_emergent=True,
        tree_search_depth=3,
        # Additional configurations
        ethical_config={
            'safety_threshold': 0.8,
            'content_filtering': True,
            'bias_detection': True
        },
        tree_config={
            'beam_width': 5,
            'max_steps_per_thought': 3
        }
    )

    # Initialize model
    model = VishwamAI(config)
    
    # Example input
    input_text = "In this paper, we propose a novel approach to"
    tokenizer = get_tokenizer()  # Assume we have a tokenizer
    input_ids = tokenizer(input_text, return_tensors='pt')['input_ids']
    
    print("Generating with standard autoregressive mode...")
    output_standard = model.generate(
        input_ids=input_ids,
        max_length=100,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2
    )
    print(tokenizer.decode(output_standard[0]))
    
    print("\nGenerating with Tree of Thoughts search...")
    output_tot = model.generate(
        input_ids=input_ids,
        max_length=100,
        temperature=0.7,
        use_tree_search=True,
        num_return_sequences=3,
        reward_config={
            'coherence_weight': 0.4,
            'novelty_weight': 0.3,
            'relevance_weight': 0.3
        }
    )
    
    print("\nMultiple completions from Tree of Thoughts:")
    for i, sequence in enumerate(output_tot):
        print(f"\nCompletion {i+1}:")
        print(tokenizer.decode(sequence))

def demonstrate_advanced_features(model, input_text):
    """Demonstrate advanced model features like MoE and ethical framework."""
    tokenizer = get_tokenizer()
    input_ids = tokenizer(input_text, return_tensors='pt')['input_ids']
    
    # Get model outputs with all features
    outputs = model.forward(
        input_ids=input_ids,
        attention_mask=None,
        use_tree_search=False
    )
    
    # Display various model outputs
    print("\nModel Analysis:")
    print("--------------")
    
    if 'ethical_scores' in outputs:
        print("\nEthical Analysis:")
        ethical_scores = outputs['ethical_scores']
        print(f"Safety Score: {ethical_scores['safety']:.2f}")
        print(f"Bias Score: {ethical_scores['bias']:.2f}")
    
    if 'emergent_patterns' in outputs:
        print("\nEmergent Behavior Analysis:")
        patterns = outputs['emergent_patterns']
        print(f"Detected Patterns: {patterns['detected_patterns']}")
        print(f"Complexity Score: {patterns['complexity_score']:.2f}")
    
    if 'memory_states' in outputs:
        print("\nMemory Analysis:")
        memory = outputs['memory_states']
        print(f"Memory Usage: {memory['usage_stats']}")
        print(f"Memory Retrieval Score: {memory['retrieval_score']:.2f}")

def get_tokenizer():
    """
    Helper function to get the tokenizer.
    In practice, you would initialize this with your actual tokenizer.
    """
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("gpt2")

if __name__ == "__main__":
    # Run the demonstration
    main()
    
    # Note: The above example assumes you have:
    # 1. Trained the model or loaded pretrained weights
    # 2. Set up the tokenizer
    # 3. Have sufficient GPU memory for the model size
    # 
    # For production use, you would want to:
    # 1. Add error handling
    # 2. Implement proper logging
    # 3. Add model parameter validation
    # 4. Implement proper resource management
    # 5. Add progress callbacks for long generations
