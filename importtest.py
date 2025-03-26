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
            'from vishwamai.layers.attention import FlashAttention',
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
        'Multimodal Dependencies': [
            'import PIL',
            'from PIL import Image',
            'import torchvision',
            'import timm',
            'from transformers import CLIPProcessor, CLIPModel',
            'import cv2',
            'import albumentations',
            'import kornia',
            'from vishwamai.multimodal.vision import ViTEncoder, CLIPAdapter',
            'from vishwamai.multimodal.fusion import CrossAttentionFuser, MultimodalProjector',
            'from vishwamai.multimodal.processor import ImageProcessor, MultimodalBatchProcessor'
        ],
        'TPU Kernels': [
            'from vishwamai.kernels.kernel import fp8_gemm_optimized, act_quant',
            'from vishwamai.kernels.fp8_cast_bf16 import bf16_cast_to_fp8',
            'from vishwamai.kernels.activation import gelu_approx, silu_optimized',
            'from vishwamai.kernels.quantization import dynamic_quant, static_quant',
            'from vishwamai.kernels.tensor_parallel import shard_params, all_gather, all_reduce',
            'from vishwamai.kernels.sparse import sparse_gemm, sparse_attention',
            'from vishwamai.kernels.moe_dispatch import load_balance_loss, compute_routing_prob'
        ],
        'TPU Optimized Layers': [
            'from vishwamai.layers.layers import TPUGEMMLinear, TPULayerNorm, TPUMultiHeadAttention',
            'from vishwamai.layers.moe import TPUMoELayer, TPUSparseMoEDispatch',
            'from vishwamai.layers.rotary import TPURotaryEmbedding, apply_rotary_pos_emb',
            'from vishwamai.layers.activation import GELUActivation, SwiGLUActivation',
            'from vishwamai.layers.normalization import RMSNorm, AdaNorm',
            'from vishwamai.layers.attention import FlashAttention'
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
            from vishwamai.multimodal.image_processor import ImageProcessor
            processor = ImageProcessor(image_size=224)
            dummy_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            processed = processor(dummy_image)
            print(f"✓ Image processor - Output shape: {processed.shape}")
        except Exception as e:
            print(f"✗ Image processor failed: {str(e)}")
            
        # Test vision encoder
        print("2. Testing vision encoder...")
        try:
            from vishwamai.multimodal.vision import ViTEncoder
            
            rng = jax.random.PRNGKey(0)
            encoder = ViTEncoder(
                hidden_dim=768,
                num_layers=2,
                num_heads=12,
                mlp_dim=3072,
                patch_size=16,
                image_size=224,
                dropout_rate=0.0,
                attention_dropout_rate=0.0,
                dtype=jnp.float32
            )
            
            dummy_image = jnp.ones((1, 224, 224, 3))
            variables = encoder.init(rng, dummy_image)
            output = encoder.apply(variables, dummy_image)
            print(f"✓ Vision encoder - Output shape: {output['last_hidden_state'].shape}")
        except Exception as e:
            print(f"✗ Vision encoder failed: {str(e)}")
            
        # Test multimodal fusion
        print("3. Testing multimodal fusion...")
        try:
            from vishwamai.multimodal.fusion import CrossAttentionFuser
            
            fuser = CrossAttentionFuser(
                hidden_dim=768,
                num_heads=12
            )
            
            vision_feats = jnp.ones((1, 196, 768))
            text_feats = jnp.ones((1, 32, 768))
            variables = fuser.init(rng, vision_feats, text_feats, deterministic=True)
            output = fuser.apply(variables, vision_feats, text_feats, deterministic=True)
            print(f"✓ Multimodal fusion - Output shape: {output['vision_output'].shape}")
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
            # Use smaller matrices compatible with quantization
            batch_size = 32
            x = jnp.ones((batch_size, 64), dtype=jnp.float32)
            y = jnp.ones((64, batch_size), dtype=jnp.float32)
            
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
                from vishwamai.layers.layers import fp8_gemm_optimized
                
                # Warmup
                _ = fp8_gemm_optimized(x, y)
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), _)
                
                # Measure optimized GEMM
                start = time.time()
                for _ in range(10):
                    result = fp8_gemm_optimized(x, y)
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
            x = jnp.ones((1024, 1024), dtype=jnp.float32)
            
            # Standard GELU
            start = time.time()
            _ = jax.nn.gelu(x)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), _)
            std_time = time.time() - start
            
            # Try optimized activation if available
            try:
                from vishwamai.kernels.ops.activation import gelu_approx
                
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

def test_tpu_kernels():
    """Test the newly implemented TPU kernel capabilities."""
    try:
        import jax
        import jax.numpy as jnp
        import time
        import numpy as np
        
        print("\nTesting TPU Kernel Implementations:")
        
        # Test TPUGEMMKernel
        print("1. Testing TPUGEMMKernel...")
        try:
            from vishwamai.kernels.tpu import TPUGEMMKernel
            
            # Create test data
            batch_size = 32
            m, n, k = 128, 128, 128
            x = jnp.ones((batch_size, m, k), dtype=jnp.float32)
            y = jnp.ones((batch_size, k, n), dtype=jnp.float32)
            
            # Initialize TPU GEMM kernel
            tpu_gemm = TPUGEMMKernel(block_size=128, use_bfloat16=True)
            
            # Warmup
            _ = tpu_gemm(x, y)
            jax.block_until_ready(_)
            
            # Benchmark
            start = time.time()
            result = tpu_gemm(x, y)
            jax.block_until_ready(result)
            elapsed = time.time() - start
            
            print(f"✓ TPUGEMMKernel - Execution time: {elapsed:.4f}s, Output shape: {result.shape}")
            
            # Compare with standard matmul
            std_result = jnp.matmul(x, y)
            accuracy = jnp.allclose(result, std_result, rtol=1e-2, atol=1e-2)
            print(f"  Accuracy check: {'Passed' if accuracy else 'Failed'}")
            
        except ImportError as e:
            print(f"✗ TPUGEMMKernel import failed: {str(e)}")
        except Exception as e:
            print(f"✗ TPUGEMMKernel test failed: {str(e)}")
            
        # Test TPULayerNormKernel
        print("2. Testing TPULayerNormKernel...")
        try:
            from vishwamai.kernels.tpu import TPULayerNormKernel
            
            # Create test data
            batch_size = 32
            seq_len = 128
            hidden_dim = 768
            x = jnp.ones((batch_size, seq_len, hidden_dim), dtype=jnp.float32)
            weight = jnp.ones(hidden_dim, dtype=jnp.float32)
            bias = jnp.zeros(hidden_dim, dtype=jnp.float32)
            
            # Initialize TPU LayerNorm kernel
            layernorm = TPULayerNormKernel(dim=hidden_dim, use_bfloat16=True)
            
            # Warmup
            _ = layernorm(x, weight, bias)
            jax.block_until_ready(_)
            
            # Benchmark
            start = time.time()
            result = layernorm(x, weight, bias)
            jax.block_until_ready(result)
            elapsed = time.time() - start
            
            print(f"✓ TPULayerNormKernel - Execution time: {elapsed:.4f}s, Output shape: {result.shape}")
            
            # Compare with standard layernorm
            def standard_layernorm(x, weight, bias, eps=1e-5):
                mean = jnp.mean(x, axis=-1, keepdims=True)
                var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
                x_norm = (x - mean) / jnp.sqrt(var + eps)
                return x_norm * weight + bias
                
            std_result = standard_layernorm(x, weight, bias)
            accuracy = jnp.allclose(result, std_result, rtol=1e-2, atol=1e-2)
            print(f"  Accuracy check: {'Passed' if accuracy else 'Failed'}")
            
        except ImportError as e:
            print(f"✗ TPULayerNormKernel import failed: {str(e)}")
        except Exception as e:
            print(f"✗ TPULayerNormKernel test failed: {str(e)}")
            
        # Test TPUAttentionKernel
        print("3. Testing TPUAttentionKernel...")
        try:
            from vishwamai.kernels.tpu import TPUAttentionKernel
            
            # Create test data
            batch_size = 2
            num_heads = 12
            seq_len = 128
            head_dim = 64
            
            query = jnp.ones((batch_size, num_heads, seq_len, head_dim), dtype=jnp.float32)
            key = jnp.ones((batch_size, num_heads, seq_len, head_dim), dtype=jnp.float32)
            value = jnp.ones((batch_size, num_heads, seq_len, head_dim), dtype=jnp.float32)
            
            # Initialize TPU Attention kernel
            attention = TPUAttentionKernel(use_flash=False, use_bfloat16=True)
            
            # Warmup
            _ = attention(query, key, value)
            jax.block_until_ready(_)
            
            # Benchmark
            start = time.time()
            result = attention(query, key, value)
            jax.block_until_ready(result)
            elapsed = time.time() - start
            
            print(f"✓ TPUAttentionKernel - Execution time: {elapsed:.4f}s, Output shape: {result.shape}")
            
        except ImportError as e:
            print(f"✗ TPUAttentionKernel import failed: {str(e)}")
        except Exception as e:
            print(f"✗ TPUAttentionKernel test failed: {str(e)}")
            
        # Test TPUFlashAttention
        print("4. Testing TPUFlashAttention...")
        try:
            from vishwamai.kernels.tpu import TPUFlashAttention
            
            # Create test data
            batch_size = 2
            num_heads = 12
            seq_len = 128
            head_dim = 64
            
            query = jnp.ones((batch_size, num_heads, seq_len, head_dim), dtype=jnp.float32)
            key = jnp.ones((batch_size, num_heads, seq_len, head_dim), dtype=jnp.float32)
            value = jnp.ones((batch_size, num_heads, seq_len, head_dim), dtype=jnp.float32)
            
            # Initialize TPU Flash Attention
            flash_attention = TPUFlashAttention(block_size=128, use_bfloat16=True)
            
            # Warmup
            _ = flash_attention(query, key, value)
            jax.block_until_ready(_)
            
            # Benchmark
            start = time.time()
            result = flash_attention(query, key, value)
            jax.block_until_ready(result)
            elapsed = time.time() - start
            
            print(f"✓ TPUFlashAttention - Execution time: {elapsed:.4f}s, Output shape: {result.shape}")
            
            # Compare with standard attention for small sequence
            def standard_attention(q, k, v, scale=None):
                if scale is None:
                    scale = 1.0 / jnp.sqrt(q.shape[-1])
                scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
                attention_weights = jax.nn.softmax(scores, axis=-1)
                return jnp.einsum('bhqk,bhkd->bhqd', attention_weights, v)
                
            std_result = standard_attention(query, key, value)
            accuracy = jnp.allclose(result, std_result, rtol=1e-2, atol=1e-2)
            print(f"  Accuracy check: {'Passed' if accuracy else 'Failed'}")
            
            # Memory efficiency test with longer sequence
            try:
                long_seq_len = 2048
                query_long = jnp.ones((batch_size, num_heads, long_seq_len, head_dim), dtype=jnp.float32)
                key_long = jnp.ones((batch_size, num_heads, long_seq_len, head_dim), dtype=jnp.float32)
                value_long = jnp.ones((batch_size, num_heads, long_seq_len, head_dim), dtype=jnp.float32)
                
                result_long = flash_attention(query_long, key_long, value_long)
                jax.block_until_ready(result_long)
                print(f"  Long sequence test (length={long_seq_len}): Passed")
            except Exception as e:
                print(f"  Long sequence test (length={long_seq_len}): Failed - {str(e)}")
            
        except ImportError as e:
            print(f"✗ TPUFlashAttention import failed: {str(e)}")
        except Exception as e:
            print(f"✗ TPUFlashAttention test failed: {str(e)}")
            
        # Test layout optimization functions
        print("5. Testing TPU layout optimization utilities...")
        try:
            from vishwamai.kernels.tpu import optimize_tpu_layout, pad_to_tpu_multiple
            
            # Test optimize_tpu_layout
            x = jnp.ones((batch_size, seq_len, hidden_dim), dtype=jnp.float32)
            opt_x = optimize_tpu_layout(x)
            print(f"✓ optimize_tpu_layout - Input shape: {x.shape}, Output shape: {opt_x.shape}")
            
            # Test pad_to_tpu_multiple
            x_padded = pad_to_tpu_multiple(x, multiple=128)
            print(f"✓ pad_to_tpu_multiple - Input shape: {x.shape}, Output shape: {x_padded.shape}")
            
        except ImportError as e:
            print(f"✗ TPU layout optimization utilities import failed: {str(e)}")
        except Exception as e:
            print(f"✗ TPU layout optimization utilities test failed: {str(e)}")
            
        print("\nTPU kernel tests completed")
    except Exception as e:
        print(f"\nTPU kernel tests failed with error: {str(e)}")

def test_tpu_kernel_imports():
    """Add specific tests for TPU kernel imports."""
    print("\nTesting TPU Kernel Imports:")
    imports = [
        "from vishwamai.kernels.tpu import TPUGEMMKernel",
        "from vishwamai.kernels.tpu import TPUAttentionKernel",
        "from vishwamai.kernels.tpu import TPULayerNormKernel",
        "from vishwamai.kernels.tpu import TPUFlashAttention",
        "from vishwamai.kernels.tpu import optimize_tpu_layout",
        "from vishwamai.kernels.tpu import pad_to_tpu_multiple",
        "from vishwamai.kernels.tpu import get_optimal_tpu_layout",
        "from vishwamai.kernels.tpu import tpu_custom_call",
        "from vishwamai.kernels.tpu import compile_tpu_kernel",
        "from vishwamai.kernels.tpu.tpu_custom_call import tpu_custom_call_lowering",
    ]
    
    successful = 0
    for import_stmt in imports:
        try:
            exec(import_stmt)
            print(f"✓ {import_stmt}")
            successful += 1
        except ImportError as e:
            print(f"✗ {import_stmt} - Error: {str(e)}")
        except Exception as e:
            print(f"! {import_stmt} - Unexpected error: {str(e)}")
    
    print(f"\nTPU Kernel Import Summary: {successful}/{len(imports)} successful")

if __name__ == "__main__":
    test_imports()
    test_multimodal_functionality()
    test_kernel_performance()
    test_tpu_kernel_imports()
    test_tpu_kernels()