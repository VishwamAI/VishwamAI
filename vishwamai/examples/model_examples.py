"""
Example uses of the VishwamAI model demonstrating different capabilities.
Each example highlights specific components and their benefits.
"""

import torch
from ..model import VishwamAIModel
from ..neural_memory import ReasoningMemoryTransformer
from ..tree_of_thoughts import TreeOfThoughts
from ..cache_augmentation import DifferentiableCacheAugmentation

def mathematical_reasoning_example():
    """Example showing enhanced mathematical reasoning capabilities."""
    model = VishwamAIModel.from_pretrained("kasinadhsarma/vishwamai-model")
    
    problem = """
    Solve this step by step:
    A company started with 100 employees. In the first quarter, they hired 30% more employees.
    In the second quarter, they had to lay off 20% of the workforce.
    In the third quarter, they hired 25% more employees.
    How many employees does the company have now?
    """
    
    # Generate with tree of thoughts for structured reasoning
    response = model.generate_with_tree(
        problem,
        max_length=512,
        beam_width=4,
        temperature=0.7
    )
    
    print("Mathematical Reasoning Example:")
    print(f"Problem: {problem}")
    print(f"Solution: {response}\n")

def scientific_analysis_example():
    """Example demonstrating scientific paper analysis with memory."""
    model = VishwamAIModel.from_pretrained("kasinadhsarma/vishwamai-model")
    
    paper_abstract = """
    Recent advances in quantum computing have demonstrated the potential for solving optimization
    problems exponentially faster than classical computers. This paper presents a novel quantum
    algorithm that achieves quadratic speedup for graph coloring problems...
    """
    
    query = "What are the key innovations and limitations of the proposed algorithm?"
    
    # Use neural memory for detailed analysis
    response = model.generate_with_memory(
        query,
        context=paper_abstract,
        memory_size=2048,
        num_memory_layers=3
    )
    
    print("Scientific Analysis Example:")
    print(f"Abstract: {paper_abstract}")
    print(f"Query: {query}")
    print(f"Analysis: {response}\n")

def code_generation_example():
    """Example showing code generation with cache augmentation."""
    model = VishwamAIModel.from_pretrained("kasinadhsarma/vishwamai-model")
    
    prompt = """
    Write a Python function that implements a Red-Black Tree with the following operations:
    - insert
    - delete
    - search
    Include proper balancing and color adjustments.
    """
    
    # Use cache for consistent code generation
    response = model.generate_with_cache(
        prompt,
        max_length=1024,
        temperature=0.3,
        cache_size=65536
    )
    
    print("Code Generation Example:")
    print(f"Prompt: {prompt}")
    print(f"Generated Code: {response}\n")

def multi_step_reasoning_example():
    """Example demonstrating complex reasoning with all components."""
    model = VishwamAIModel.from_pretrained("kasinadhsarma/vishwamai-model")
    
    scenario = """
    Design a system to optimize traffic flow in a smart city with the following constraints:
    1. Multiple intersections with varying traffic patterns
    2. Emergency vehicle priority
    3. Pedestrian safety
    4. Public transportation schedules
    5. Real-time adaptation to accidents or road works
    
    Provide a detailed solution considering all aspects.
    """
    
    # Use all enhancement components
    response = model.generate_enhanced(
        scenario,
        use_memory=True,
        use_tree=True,
        use_cache=True,
        max_length=2048,
        temperature=0.7
    )
    
    print("Multi-step Reasoning Example:")
    print(f"Scenario: {scenario}")
    print(f"Solution: {response}\n")

def research_synthesis_example():
    """Example showing research synthesis across multiple papers."""
    model = VishwamAIModel.from_pretrained("kasinadhsarma/vishwamai-model")
    
    papers = [
        """Paper 1: Recent advances in transformer architectures have shown significant 
        improvements in natural language understanding...""",
        """Paper 2: Self-attention mechanisms can be optimized through sparse 
        implementations that reduce computational complexity...""",
        """Paper 3: Memory-efficient transformers demonstrate comparable performance 
        while using only 20% of the parameters..."""
    ]
    
    query = "Synthesize the key trends and future directions in transformer optimization."
    
    # Use memory and tree for complex synthesis
    response = model.generate_research_synthesis(
        query,
        papers=papers,
        max_length=1024,
        temperature=0.6
    )
    
    print("Research Synthesis Example:")
    print("Papers:", *papers, sep="\n")
    print(f"Query: {query}")
    print(f"Synthesis: {response}\n")

def batch_processing_example():
    """Example demonstrating efficient batch processing."""
    model = VishwamAIModel.from_pretrained("kasinadhsarma/vishwamai-model")
    
    queries = [
        "Explain quantum entanglement.",
        "What is the complexity of quicksort?",
        "How do neural networks learn?",
        "Describe the greenhouse effect."
    ]
    
    # Process multiple queries efficiently
    responses = model.batch_generate(
        queries,
        max_length=256,
        batch_size=4,
        use_cache=True
    )
    
    print("Batch Processing Example:")
    for query, response in zip(queries, responses):
        print(f"Query: {query}")
        print(f"Response: {response}\n")

if __name__ == "__main__":
    print("Running VishwamAI Model Examples...")
    
    examples = [
        mathematical_reasoning_example,
        scientific_analysis_example,
        code_generation_example,
        multi_step_reasoning_example,
        research_synthesis_example,
        batch_processing_example
    ]
    
    for example in examples:
        try:
            example()
            print("-" * 80 + "\n")
        except Exception as e:
            print(f"Error in {example.__name__}: {str(e)}\n")
