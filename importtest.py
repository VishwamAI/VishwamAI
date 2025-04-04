#!/usr/bin/env python3
"""Import test suite for VishwamAI components."""

import unittest
import sys


class TestVishwamAIImports(unittest.TestCase):
    """Test suite to verify all VishwamAI component imports."""

    def test_core_kernels(self):
        """Test kernel-related imports."""
        try:
            from vishwamai.kernels import (
                TPUGEMMLinear,
                TPULayerNorm,
                TPUMultiHeadAttention,
                TPUMoELayer,
                FlashAttention,
                KernelConfig,
                MemoryLayout
            )
            self.assertTrue(True, "Kernel imports successful")
        except ImportError as e:
            self.fail(f"Kernel import failed: {e}")

    def test_layers(self):
        """Test layer-related imports."""
        try:
            from vishwamai.layers.layers import (
                TPUGEMMLinear,
                TPULayerNorm,
                TPUMultiHeadAttention,
                TPUMoELayer
            )
            from vishwamai.layers.attention import FlashAttention
            self.assertTrue(True, "Layer imports successful")
        except ImportError as e:
            self.fail(f"Layer import failed: {e}")

    def test_multimodal(self):
        """Test multimodal-related imports."""
        try:
            from vishwamai.multimodal import (
                MultimodalProcessor,
                ImageProcessor,
                MultimodalBatchProcessor,
                ImageCaptioningPipeline,
                VisualQuestionAnswering,
                AudioCaptioningPipeline,
                MultimodalChatPipeline,
                MultilingualPipeline
            )
            from vishwamai.multimodal.config import (
                VisionConfig,
                AudioConfig,
                FusionConfig,
                MultimodalConfig
            )
            self.assertTrue(True, "Multimodal imports successful")
        except ImportError as e:
            self.fail(f"Multimodal import failed: {e}")

    def test_thoughts(self):
        """Test reasoning/thoughts-related imports."""
        try:
            from vishwamai.thoughts.tot import TreeOfThoughts
            from vishwamai.thoughts.cot import validate_reasoning
            self.assertTrue(True, "Thoughts imports successful")
        except ImportError as e:
            self.fail(f"Thoughts import failed: {e}")

    def test_model(self):
        """Test core model imports."""
        try:
            from vishwamai.model import (
                VishwamAI,
                VishwamAIConfig,
                TransformerBlock,
                FeedForward
            )
            self.assertTrue(True, "Model imports successful")
        except ImportError as e:
            self.fail(f"Model import failed: {e}")

    def test_development_utils(self):
        """Test development utilities."""
        try:
            from vishwamai.profiler import TPUProfiler
            from vishwamai.pipeline import VishwamAIPipeline
            from vishwamai.device_mesh import TPUMeshContext
            from vishwamai.logger import DuckDBLogger
            self.assertTrue(True, "Development utilities imports successful")
        except ImportError as e:
            self.fail(f"Development utilities import failed: {e}")


def print_test_summary(result):
    """Print a summary of test results."""
    print("\nImport Test Summary:")
    print("=" * 50)
    print(f"Core Dependencies: {result.testsRun}/6 successful")
    if result.wasSuccessful():
        print("\nAll imports successful! ✅")
    else:
        print("\nSome imports failed. Check error messages above. ❌")
    print("=" * 50)


if __name__ == "__main__":
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestVishwamAIImports)
    
    # Run tests and capture results
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print_test_summary(result)
    
    # Set exit code based on test success
    sys.exit(not result.wasSuccessful())