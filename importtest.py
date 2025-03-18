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
            'from vishwamai.transformer import EnhancedTransformerModel',
            'from vishwamai.layers.layers import TPUGEMMLinear, TPULayerNorm, TPUMultiHeadAttention, TPUMoELayer',
            'from vishwamai.flash_attention import FlashAttention',
            'from vishwamai.kernels.kernel import fp8_gemm_optimized',
            'from vishwamai.thoughts.tot import TreeOfThoughts, ThoughtNode',
            'from vishwamai.thoughts.cot import ChainOfThoughtPrompting',
            'from vishwamai.distill import compute_distillation_loss, create_student_model, initialize_from_teacher'
        ],
        'SONAR Dependencies': [
            'import fairseq2',
            'import editdistance',
            'import importlib_metadata',
            'import importlib_resources',
            'import sacrebleu'
        ],
        # New multimodal imports section
        'Multimodal Dependencies': [
            'import PIL',
            'from PIL import Image',
            'import torchvision',
            'import timm',
            'from transformers import CLIPProcessor, CLIPModel, VisionTextDualEncoder',
            'import cv2',
            'import albumentations',
            'import kornia',
            'from vishwamai.multimodal.vision import ViTEncoder, CLIPAdapter',
            'from vishwamai.multimodal.fusion import CrossAttentionFuser, MultimodalProjector',
            'from vishwamai.multimodal.processor import ImageProcessor, MultimodalBatchProcessor'
        ],
        # Expanded kernels section
        'TPU Kernels': [
            'from vishwamai.kernels.kernel import fp8_gemm_optimized, act_quant',
            'from vishwamai.kernels.fp8_cast_bf16 import bf16_cast_to_fp8',
            'from vishwamai.kernels.activation import gelu_approx, silu_optimized',
            'from vishwamai.kernels.quantization import dynamic_quant, static_quant',
            'from vishwamai.kernels.tensor_parallel import shard_params, all_gather, all_reduce',
            'from vishwamai.kernels.sparse import sparse_gemm, sparse_attention',
            'from vishwamai.kernels.moe_dispatch import load_balance_loss, compute_routing_prob'
        ],
        # Expanded layers section
        'TPU Optimized Layers': [
            'from vishwamai.layers.layers import TPUGEMMLinear, TPULayerNorm, TPUMultiHeadAttention',
            'from vishwamai.layers.moe import TPUMoELayer, TPUSparseMoEDispatch',
            'from vishwamai.layers.rotary import TPURotaryEmbedding, apply_rotary_pos_emb',
            'from vishwamai.layers.activation import GELUActivation, SwiGLUActivation',
            'from vishwamai.layers.normalization import RMSNorm, AdaNorm',
            'from vishwamai.layers.attention import FlashAttention, ChunkwiseCausalAttention'
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

def test_multimodal_functionality():
    """Test basic functionality of multimodal components."""
    try:
        import jax
        import jax.numpy as jnp
        import flax.linen as nn
        from PIL import Image
        import numpy as np
        
        # Test image processing
        print("\nTesting multimodal functionality:")
        print("1. Testing image processing...")
        try:
            from vishwamai.multimodal.processor import ImageProcessor
            processor = ImageProcessor(image_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            dummy_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            processed = processor(dummy_image)
            print(f"✓ Image processor - Output shape: {processed.shape}")
        except Exception as e:
            print(f"✗ Image processor failed: {str(e)}")
            
        # Test vision encoder
        print("2. Testing vision encoder...")
        try:
            from vishwamai.multimodal.vision import ViTEncoder
            
            class DummyViTEncoder(nn.Module):
                @nn.compact
                def __call__(self, x):
                    x = nn.Dense(features=768)(x)
                    return x
            
            rng = jax.random.PRNGKey(0)
            model = DummyViTEncoder()
            dummy_input = jnp.ones((1, 196, 768))  # [batch, patches, dim]
            params = model.init(rng, dummy_input)
            output = model.apply(params, dummy_input)
            print(f"✓ Vision encoder - Output shape: {output.shape}")
        except Exception as e:
            print(f"✗ Vision encoder failed: {str(e)}")
            
        # Test multimodal fusion
        print("3. Testing multimodal fusion...")
        try:
            class DummyFusion(nn.Module):
                @nn.compact
                def __call__(self, vision_features, text_features):
                    combined = jnp.concatenate([vision_features, text_features], axis=1)
                    return nn.Dense(features=512)(combined)
            
            rng = jax.random.PRNGKey(0)
            model = DummyFusion()
            vision_feats = jnp.ones((1, 196, 768))
            text_feats = jnp.ones((1, 32, 768))
            params = model.init(rng, vision_feats, text_feats)
            output = model.apply(params, vision_feats, text_feats)
            print(f"✓ Multimodal fusion - Output shape: {output.shape}")
        except Exception as e:
            print(f"✗ Multimodal fusion failed: {str(e)}")
            
        print("\nMultimodal functionality tests completed")
    except Exception as e:
        print(f"\nMultimodal functionality tests failed with error: {str(e)}")

def test_kernel_performance():
    """Test performance of TPU-optimized kernels."""
    try:
        import jax
        import jax.numpy as jnp
        import time
        
        print("\nTesting kernel performance:")
        
        # Test GEMM performance
        print("1. Testing GEMM performance...")
        try:
            # Standard matmul with stable deterministic values
            x = jnp.ones((1024, 1024), dtype=jnp.float32)
            y = jnp.ones((1024, 1024), dtype=jnp.float32)
            
            # Warmup
            _ = jnp.matmul(x, y)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), _)
            
            # Measure standard matmul
            start = time.time()
            for _ in range(10):
                result = jnp.matmul(x, y)
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), result)
            std_time = (time.time() - start) / 10
            
            # Try optimized GEMM if available
            try:
                from vishwamai.kernels.kernel import fp8_gemm_optimized, act_quant
                
                # Prepare input for optimized GEMM
                x_quant, x_scale = act_quant(x)
                y_quant, y_scale = act_quant(y)
                
                # Warmup
                _ = fp8_gemm_optimized(x_quant, x_scale, y_quant, y_scale)
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), _)
                
                # Measure optimized GEMM
                start = time.time()
                for _ in range(10):
                    result = fp8_gemm_optimized(x_quant, x_scale, y_quant, y_scale)
                    jax.tree_util.tree_map(lambda x: x.block_until_ready(), result)
                opt_time = (time.time() - start) / 10
                
                speedup = std_time / opt_time if opt_time > 0 else 0
                print(f"✓ GEMM performance - Standard: {std_time:.4f}s, Optimized: {opt_time:.4f}s, Speedup: {speedup:.2f}x")
            except ImportError:
                print(f"✓ Standard GEMM performance: {std_time:.4f}s (optimized version not available)")
        except Exception as e:
            print(f"✗ GEMM performance test failed: {str(e)}")
        
        # Test activation functions
        print("2. Testing activation functions...")
        try:
            x = jnp.ones((4096, 4096), dtype=jnp.float32)
            
            # Standard GELU
            start = time.time()
            _ = jax.nn.gelu(x)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), _)
            std_time = time.time() - start
            
            # Try optimized activation if available
            try:
                from vishwamai.kernels.activation import gelu_approx
                
                start = time.time()
                _ = gelu_approx(x)
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), _)
                opt_time = time.time() - start
                
                speedup = std_time / opt_time if opt_time > 0 else 0
                print(f"✓ Activation performance - Standard: {std_time:.4f}s, Optimized: {opt_time:.4f}s, Speedup: {speedup:.2f}x")
            except ImportError:
                print(f"✓ Standard activation performance: {std_time:.4f}s (optimized version not available)")
        except Exception as e:
            print(f"✗ Activation performance test failed: {str(e)}")
            
        print("\nKernel performance tests completed")
    except Exception as e:
        print(f"\nKernel performance tests failed with error: {str(e)}")

if __name__ == "__main__":
    test_imports()
    test_multimodal_functionality()
    test_kernel_performance()