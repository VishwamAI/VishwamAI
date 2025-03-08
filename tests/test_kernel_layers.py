# /home/kasinadhsarma/VishwamAI/tests/test_kernel_layers.py
import unittest
import torch
import triton
from vishwamai.models.kernel_layers import (
    HardwareCapabilityDetector,
    KernelTransformer,
    OptimizedLayerNorm,
    FeedForward
)

class TestGPUOptimizations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.capabilities = HardwareCapabilityDetector.get_gpu_capabilities()
        cls.has_gpu = cls.capabilities['has_gpu']
        cls.has_tensor_cores = cls.capabilities['has_tensor_cores']
        
    def test_hardware_detection(self):
        """Test GPU capability detection"""
        self.assertIsInstance(self.capabilities, dict)
        self.assertIn('has_gpu', self.capabilities)
        self.assertIn('gpu_count', self.capabilities)
        self.assertIn('has_tensor_cores', self.capabilities)
        
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_triton_kernel_execution(self):
        """Test Triton kernel execution for LayerNorm"""
        batch_size, seq_len, dim = 32, 128, 512
        x = torch.randn(batch_size, seq_len, dim).cuda()
        layer_norm = OptimizedLayerNorm(dim).cuda()
        
        # Enable Triton optimization
        layer_norm.use_triton = True
        output = layer_norm(x)
        
        self.assertEqual(output.shape, x.shape)
        self.assertTrue(output.is_cuda)
        
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gelu_optimization(self):
        """Test GELU Triton optimization"""
        batch_size, seq_len, dim = 32, 128, 512
        x = torch.randn(batch_size, seq_len, dim).cuda()
        ff = FeedForward(dim, dim * 4).cuda()
        
        # Test with half precision to trigger Triton optimization
        x = x.half()
        ff = ff.half()
        output = ff(x)
        
        self.assertEqual(output.shape, x.shape)
        self.assertTrue(output.is_cuda)
        self.assertEqual(output.dtype, torch.float16)
        
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_memory_optimization(self):
        """Test memory optimization hooks"""
        model = KernelTransformer(
            vocab_size=32000,
            embed_dim=512,
            num_layers=6,
            num_heads=8,
            ff_dim=2048
        ).cuda()
        
        # Test with different batch sizes
        inputs = torch.randint(0, 32000, (16, 128)).cuda()
        start_mem = torch.cuda.memory_allocated()
        _ = model(inputs)
        end_mem = torch.cuda.memory_allocated()
        
        # Memory should be managed efficiently
        self.assertLess(end_mem - start_mem, 1e9)  # Less than 1GB increase
        
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available") 
    def test_tensor_cores_usage(self):
        """Test TensorCore utilization when available"""
        if self.has_tensor_cores:
            # Create model with tensor core compatible dimensions
            model = KernelTransformer(
                vocab_size=32000,
                embed_dim=512,  # Multiple of 8 for tensor cores
                num_layers=6,
                num_heads=8,
                ff_dim=2048
            ).cuda()
            
            # Check if tensor cores are enabled
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
            
            # Test with half precision
            inputs = torch.randint(0, 32000, (16, 128)).cuda()
            model = model.half()
            outputs = model(inputs)
            
            self.assertEqual(outputs.dtype, torch.float16)
            
if __name__ == '__main__':
    unittest.main()