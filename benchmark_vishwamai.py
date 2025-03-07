#!/usr/bin/env python3
"""
Benchmark script for VishwamAI with Tree of Thoughts (ToT) integration
and TPU optimization.

This script demonstrates how to use VishwamAI's advanced reasoning capabilities
with memory-efficient TPU execution.
"""

import os
import time
import argparse
import json
import logging
from tqdm import tqdm
import numpy as np

import jax
import jax.numpy as jnp

from vishwamai.model import VishwamAIModel, ModelConfig
from vishwamai.tokenizer import VishwamAITokenizer
from vishwamai.tot import TreeOfThoughts
from vishwamai.generate import GenerationConfig, generate
from vishwamai.integration import ToTModelIntegrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Example reasoning tasks for benchmarking
BENCHMARK_TASKS = {
    "math": [
        "Solve step-by-step: If a triangle has sides of length 5, 12, and 13, what is its area?",
        "Calculate step-by-step: What is the sum of the first 100 positive integers?",
        "Find all solutions step-by-step: 3x² + 8x - 3 = 0"
    ],
    "reasoning": [
        "A farmer has 15 sheep, all but 8 died. How many sheep are still alive? Explain your reasoning.",
        "If I have a cake and cut it into 8 equal slices, then eat 3 slices, what fraction of the cake remains? Reason step-by-step.",
        "If a shirt originally costs $25 and is on sale for 20% off, and I have a coupon for an additional 10% off the sale price, how much will I pay? Show your work."
    ],
    "planning": [
        "I need to prepare dinner for five people. Create a step-by-step plan for making spaghetti with meatballs.",
        "Create a study plan for someone preparing for a final exam with only 3 days left.",
        "Outline the steps needed to create a personal budget for someone who has never made one before."
    ]
}

def memory_usage():
    """Get current memory usage in GB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / (1024 * 1024 * 1024)  # Convert to GB
        return mem
    except ImportError:
        return 0

def benchmark_generation(
    model, 
    tokenizer, 
    prompts, 
    config, 
    tot_model=None
):
    """Benchmark text generation with performance metrics."""
    results = []
    total_tokens = 0
    start_time = time.time()
    
    # Log initial memory usage
    initial_mem = memory_usage()
    logger.info(f"Initial memory usage: {initial_mem:.2f}GB")
    
    # Set up random seed for reproducibility
    rng = jax.random.PRNGKey(42)
    
    # Process in batches to better utilize TPU
    for i in range(0, len(prompts), config.batch_size):
        batch_prompts = prompts[i:i+config.batch_size]
        batch_start = time.time()
        
        # Generate text
        batch_results = generate(
            model,
            tokenizer,
            batch_prompts,
            config,
            rng=rng,
            tot_model=tot_model
        )
        
        # Track tokens generated
        for text in batch_results['generated_texts']:
            tokens = tokenizer.encode(text)
            total_tokens += len(tokens)
        
        # Record batch timing
        batch_time = time.time() - batch_start
        batch_size = len(batch_prompts)
        
        # Log batch results
        logger.info(f"Batch {i//config.batch_size + 1}: {batch_time:.2f}s for {batch_size} prompts")
        
        # Add results
        results.extend(batch_results['generated_texts'])
    
    # Calculate overall metrics
    total_time = time.time() - start_time
    throughput = total_tokens / total_time
    
    # Log final memory usage
    final_mem = memory_usage()
    mem_diff = final_mem - initial_mem
    
    metrics = {
        'total_time_seconds': total_time,
        'total_tokens': total_tokens,
        'token_throughput': throughput,
        'prompts_processed': len(prompts),
        'initial_memory_gb': initial_mem,
        'final_memory_gb': final_mem,
        'memory_difference_gb': mem_diff,
        'tpu_devices': jax.device_count()
    }
    
    return results, metrics

def main():
    parser = argparse.ArgumentParser(description='Benchmark VishwamAI with Tree of Thoughts')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model weights')
    parser.add_argument('--config-path', type=str, required=True, help='Path to model config')
    parser.add_argument('--tokenizer-path', type=str, required=True, help='Path to tokenizer')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for generation')
    parser.add_argument('--task-type', choices=['math', 'reasoning', 'planning', 'all'], default='all',
                       help='Type of benchmark tasks to run')
    parser.add_argument('--output', type=str, help='Path to save results JSON')
    parser.add_argument('--max-tokens', type=int, default=512, help='Maximum tokens to generate')
    parser.add_argument('--use-tot', action='store_true', help='Enable Tree of Thoughts reasoning')
    parser.add_argument('--tot-search', choices=['beam', 'dfs', 'bfs', 'mcts'], default='beam',
                       help='ToT search strategy')
    parser.add_argument('--use-error-correction', action='store_true', help='Enable error correction')
    parser.add_argument('--chunk-size', type=int, default=32, help='Chunk size for memory-efficient processing')
    args = parser.parse_args()

    try:
        # Print JAX device info
        logger.info(f"JAX devices: {jax.devices()}")
        logger.info(f"Number of devices: {jax.device_count()}")
        
        # Load model configuration
        logger.info(f"Loading configuration from {args.config_path}")
        with open(args.config_path) as f:
            config = ModelConfig(**json.load(f))
        
        # Print memory usage at start
        logger.info(f"Initial memory usage: {memory_usage():.2f}GB")
            
        # Initialize model
        logger.info("Initializing model...")
        model = VishwamAIModel(config)
        
        # Load model weights
        logger.info(f"Loading model weights from {args.model_path}")
        model.load_weights(args.model_path)
        
        # Initialize tokenizer
        logger.info(f"Loading tokenizer from {args.tokenizer_path}")
        tokenizer = VishwamAITokenizer.from_pretrained(args.tokenizer_path)
        
        # Initialize Tree of Thoughts if enabled
        tot_model = None
        if args.use_tot:
            logger.info("Initializing Tree of Thoughts...")
            tot_model = TreeOfThoughts(
                transformer=model,
                tokenizer=tokenizer,
                max_thoughts=5,
                max_depth=3,
                beam_width=5,
                use_tpu=True
            )
            logger.info(f"Tree of Thoughts initialized with {args.tot_search} search strategy")
        
        # Set up generation config
        gen_config = GenerationConfig(
            max_new_tokens=args.max_tokens,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            use_error_correction=args.use_error_correction,
            use_tot=args.use_tot,
            tot_search_strategy=args.tot_search
        )
        
        # Select benchmark tasks
        prompts = []
        if args.task_type == 'all':
            for task_prompts in BENCHMARK_TASKS.values():
                prompts.extend(task_prompts)
        else:
            prompts = BENCHMARK_TASKS[args.task_type]
        
        logger.info(f"Running benchmark with {len(prompts)} prompts")
        
        # Run benchmark
        results, metrics = benchmark_generation(
            model, 
            tokenizer, 
            prompts, 
            gen_config, 
            tot_model
        )
        
        # Print metrics
        logger.info("Benchmark complete")
        logger.info(f"Total time: {metrics['total_time_seconds']:.2f}s")
        logger.info(f"Tokens generated: {metrics['total_tokens']}")
        logger.info(f"Throughput: {metrics['token_throughput']:.2f} tokens/sec")
        logger.info(f"Memory usage: {metrics['memory_difference_gb']:.2f}GB")
        
        # Create output data
        output_data = {
            'prompts': prompts,
            'generated_texts': results,
            'metrics': metrics,
            'config': {
                'batch_size': args.batch_size,
                'max_tokens': args.max_tokens,
                'use_tot': args.use_tot,
                'tot_search': args.tot_search if args.use_tot else None,
                'use_error_correction': args.use_error_correction,
                'chunk_size': args.chunk_size
            }
        }
        
        # Save results if output path provided
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"Results saved to {args.output}")
        
        # Print sample outputs
        logger.info("\nSample outputs:")
        for i, (prompt, result) in enumerate(zip(prompts[:3], results[:3])):
            logger.info(f"\nPrompt {i+1}: {prompt}")
            logger.info(f"Generated: {result[:200]}...")
        
    except Exception as e:
        logger.error(f"Error during benchmark: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()