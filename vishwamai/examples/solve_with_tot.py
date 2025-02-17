import torch
from transformers import AutoModelForCausalLM
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from vishwamai.tree_of_thoughts import TreeOfThoughts, TreeConfig, RewardConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_reasoning_path(node, depth=0) -> str:
    """Format a node's reasoning path into human-readable text."""
    indent = "  " * depth
    steps = []
    
    # Add node's own reasoning
    if node.text_output:
        steps.append(f"{indent}Thought: {node.text_output}")
    
    if node.reasoning_steps:
        # Format mathematical operations
        for step in node.reasoning_steps:
            op_idx = torch.argmax(step['operation']).item()
            op_symbol = ['+', '-', '*', '/', '='][op_idx]
            num = step['number'].item()
            steps.append(f"{indent}Step: {op_symbol} {num:.2f}")
    
    # Add uncertainty measure
    steps.append(f"{indent}Confidence: {1 - node.uncertainty:.2%}")
    
    # Recursively add best child's reasoning
    if node.children:
        best_child = max(node.children, key=lambda x: x.score)
        steps.extend(format_reasoning_path(best_child, depth + 1))
    
    return "\n".join(steps)

def solve_problem(
    model: TreeOfThoughts,
    problem: str,
    tokenizer: Any,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Any]:
    """Solve a problem using Tree of Thoughts reasoning."""
    # Tokenize input
    inputs = tokenizer(problem, return_tensors="pt").to(device)
    
    # Get initial hidden states from base model
    with torch.no_grad():
        hidden_states = model.base_model(**inputs).last_hidden_state
    
    # Process through tree of thoughts
    outputs = model(hidden_states)
    
    # Get best reasoning path
    if isinstance(outputs, list):
        best_node = max(model._get_leaf_nodes(outputs), key=lambda x: x.score)
        reasoning = format_reasoning_path(best_node)
        solution = best_node.text_output if best_node.text_output else "No solution found."
    else:
        # If outputs is tensor, generate solution text
        solution = model.text_generator.generate_text(outputs[0], model.config)
        reasoning = "Direct solution without intermediate steps."
    
    return {
        "problem": problem,
        "solution": solution,
        "reasoning": reasoning,
        "confidence": 1 - best_node.uncertainty if isinstance(outputs, list) else None
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True,
                       help='Base model name or path')
    parser.add_argument('--tot_path', type=str, required=True,
                       help='Path to trained Tree of Thoughts model')
    parser.add_argument('--problems_file', type=str, required=True,
                       help='JSON file containing problems to solve')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Path to save solutions')
    parser.add_argument('--beam_width', type=int, default=4)
    parser.add_argument('--max_depth', type=int, default=3)
    parser.add_argument('--max_math_steps', type=int, default=8)
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with detailed outputs')
    
    args = parser.parse_args()
    
    # Load base model and tokenizer
    logger.info("Loading models...")
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = base_model.tokenizer
    
    # Initialize Tree of Thoughts
    tree_config = TreeConfig(
        beam_width=args.beam_width,
        max_depth=args.max_depth,
        max_math_steps=args.max_math_steps,
        intermediate_steps=True,
        debug_mode=args.debug
    )
    
    model = TreeOfThoughts.from_pretrained(
        args.tot_path,
        model=base_model
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Load problems
    logger.info("Loading problems...")
    with open(args.problems_file, 'r') as f:
        problems = json.load(f)
    
    # Solve problems
    logger.info("Solving problems...")
    solutions = []
    for problem in problems:
        logger.info(f"\nSolving problem: {problem}")
        solution = solve_problem(model, problem, tokenizer)
        solutions.append(solution)
        
        if args.debug:
            logger.info(f"Solution: {solution['solution']}")
            logger.info(f"Reasoning:\n{solution['reasoning']}")
            if solution['confidence']:
                logger.info(f"Confidence: {solution['confidence']:.2%}")
                
    # Save solutions
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(solutions, f, indent=2)
    
    logger.info(f"\nSolutions saved to {output_path}")

if __name__ == '__main__':
    main()
