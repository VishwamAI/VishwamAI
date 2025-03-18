"""Test script to validate all imports and dependencies."""

def test_imports():
    imports = {
        'Core Dependencies': [
            'import jax',
            'import jax.numpy as jnp',
            'import flax.linen as nn',
            'import optax',
            'import numpy as np',
            'import torch',
            'from transformers import AutoTokenizer',
            'from safetensors import safe_open'
        ],
        'Data Processing': [
            'import datasets',
            'import sentencepiece',
            'import tokenizers',
            'from huggingface_hub import snapshot_download'
        ],
        'Training Utilities': [
            'import wandb',
            'import duckdb',
            'import tqdm',
            'import pyarrow'
        ],
        'Memory Optimization': [
            'import einops',
            'import chex',
            'import jaxtyping',
            'import optree',
            'import orbax.checkpoint'
        ],
        'Additional Libraries': [
            'import scipy',
            'from ml_collections import ConfigDict',
            'import typing_extensions'
        ],
        'VishwamAI Modules': [
            'from vishwamai.model import VishwamAI',
            'from vishwamai.layers.layers import TPUGEMMLinear, TPULayerNorm, TPUMultiHeadAttention, TPUMoELayer',
            'from vishwamai.multimodal.encoder import MultimodalEncoder',
            'from vishwamai.flash_attention import FlashAttention',
            'from vishwamai.kernels.kernel import fp8_gemm_optimized'
        ],
        'SONAR Dependencies': [
            'import fairseq2',
            'import editdistance',
            'import importlib_metadata',
            'import importlib_resources',
            'import sacrebleu'
        ]
    }

    results = {}
    for category, import_statements in imports.items():
        print(f"\nTesting {category}:")
        category_results = []
        
        for import_stmt in import_statements:
            try:
                exec(import_stmt)
                print(f"✓ {import_stmt}")
                category_results.append((import_stmt, True))
            except ImportError as e:
                print(f"✗ {import_stmt} - Error: {str(e)}")
                category_results.append((import_stmt, False))
            except Exception as e:
                print(f"! {import_stmt} - Unexpected error: {str(e)}")
                category_results.append((import_stmt, False))
                
        results[category] = category_results
    
    # Print summary
    print("\nImport Test Summary:")
    print("-------------------")
    total = 0
    successful = 0
    
    for category, category_results in results.items():
        cat_total = len(category_results)
        cat_success = sum(1 for _, success in category_results if success)
        total += cat_total
        successful += cat_success
        print(f"{category}: {cat_success}/{cat_total} successful")
    
    print(f"\nOverall: {successful}/{total} imports successful ({(successful/total)*100:.1f}%)")
    
    return results

if __name__ == "__main__":
    test_imports()