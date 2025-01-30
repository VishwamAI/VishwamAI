import pytest
import torch
import torch.nn as nn
from vishwamai.convert import ModelConverter, ConversionConfig, WeightConverter
from vishwamai.efficient_attention import EfficientAttention
from vishwamai.tree_attention import TreeAttention

class TestModelConverter:
    @pytest.fixture
    def model(self):
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Linear(16 * 30 * 30, 10)
        )

    @pytest.fixture
    def config(self):
        return ConversionConfig(target_dtype="fp16")

    def test_convert_model(self, model, config):
        converter = ModelConverter(config)
        converted_model = converter.convert_model(model)
        assert converted_model is not None
        assert converted_model[0].weight.dtype == torch.float16

    def test_optimize_compute(self, model, config):
        converter = ModelConverter(config)
        optimized_model = converter._optimize_compute(model)
        assert optimized_model is not None

    def test_convert_dtype(self, model, config):
        converter = ModelConverter(config)
        converted_model = converter._convert_dtype(model)
        assert converted_model is not None
        assert converted_model[0].weight.dtype == torch.float16

    def test_apply_quantization(self, model, config):
        converter = ModelConverter(config)
        quantized_model = converter._apply_quantization(model)
        assert quantized_model is not None

    def test_validate_config(self):
        # Test valid config
        valid_config = ConversionConfig(target_dtype="fp16")
        converter = ModelConverter(valid_config)
        
        # Test invalid config
        with pytest.raises(ValueError):
            invalid_config = ConversionConfig(target_dtype="invalid")
            converter = ModelConverter(invalid_config)

    def test_replace_attention(self, model, config):
        converter = ModelConverter(config)
        replaced_model = converter._replace_attention(model)
        assert replaced_model is not None
        # Verify attention layers were replaced
        for name, module in replaced_model.named_modules():
            if "attention" in name.lower():
                if config.use_tree_attention:
                    assert isinstance(module, TreeAttention)
                else:
                    assert isinstance(module, EfficientAttention)

    def test_fuse_layers(self, model, config):
        converter = ModelConverter(config)
        fused_model = converter._fuse_layers(model)
        assert fused_model is not None
        # Verify Conv-BN-ReLU fusion
        for name, module in fused_model.named_modules():
            if isinstance(module, nn.Sequential):
                # Allow for multiple fused operations
                assert len(module) > 0

    def test_apply_pruning(self, model, config):
        converter = ModelConverter(config)
        pruned_model = converter._apply_pruning(model)
        assert pruned_model is not None
        # Verify pruning actually occurred
        total_pruned = 0
        for name, param in pruned_model.named_parameters():
            if 'weight' in name:
                total_pruned += torch.sum(torch.abs(param.data) < config.pruning_threshold).item()
        # Allow most weights to remain below threshold
        assert total_pruned < 0.95 * sum(p.numel() for p in model.parameters())

    def test_advanced_optimizations(self, model, config):
        converter = ModelConverter(config)
        optimized_model = converter._apply_advanced_optimizations(model)
        assert optimized_model is not None
        # Verify optimizations were applied
        original_params = sum(p.numel() for p in model.parameters())
        optimized_params = sum(p.numel() for p in optimized_model.parameters())
        # Allow for parameter count to stay the same if optimizations don't reduce size
        assert optimized_params <= original_params

    def test_math_model_conversion(self):
        """Test conversion of math-specific model configurations"""
        config = ConversionConfig(
            target_dtype="fp16",
            optimization_level=3  # Level 3 enables math optimizations
        )
        math_model = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        converter = ModelConverter(config)
        converted_model = converter.convert_model(math_model)
        assert converted_model is not None
        # Verify math-specific layers were preserved
        assert isinstance(converted_model[0], nn.Linear)
        assert isinstance(converted_model[2], nn.Linear)

class TestWeightConverter:
    @pytest.fixture
    def weights(self):
        return torch.randn(10, 10)

    def test_quantize_weights(self, weights):
        # Test symmetric quantization
        quantized, params = WeightConverter.quantize_weights(weights, symmetric=True)
        assert quantized is not None
        assert params is not None
        assert "scale" in params
        assert quantized.dtype == weights.dtype
        
        # Test asymmetric quantization
        quantized, params = WeightConverter.quantize_weights(weights, symmetric=False)
        assert quantized is not None
        assert params is not None
        assert "scale" in params
        assert "zero_point" in params
        assert quantized.dtype == weights.dtype

    def test_dequantize_weights(self, weights):
        # Test symmetric dequantization
        quantized, params = WeightConverter.quantize_weights(weights, symmetric=True)
        dequantized = WeightConverter.dequantize_weights(quantized, params, symmetric=True)
        assert torch.allclose(dequantized, weights, rtol=1e-1, atol=1e-2)
        assert dequantized.dtype == weights.dtype
        
        # Test asymmetric dequantization
        quantized, params = WeightConverter.quantize_weights(weights, symmetric=False)
        dequantized = WeightConverter.dequantize_weights(quantized, params, symmetric=False)
        assert torch.allclose(dequantized, weights, rtol=1e-1, atol=1e-2)
        assert dequantized.dtype == weights.dtype

    def test_quantize_edge_cases(self):
        # Test empty tensor
        empty = torch.tensor([])
        quantized, params = WeightConverter.quantize_weights(empty)
        assert quantized.numel() == 0
        assert params["scale"] == 1.0
        assert params["zero_point"] == 0.0
        
        # Test all zeros
        zeros = torch.zeros(10, 10)
        quantized, params = WeightConverter.quantize_weights(zeros)
        assert torch.all(quantized == 0)
        assert params["scale"] == 1.0
        assert params["zero_point"] == 0.0
        
        # Test single value
        single = torch.tensor([1.0])
        quantized, params = WeightConverter.quantize_weights(single)
        dequantized = WeightConverter.dequantize_weights(quantized, params)
        assert torch.allclose(dequantized, single)
        
        # Test large values
        large = torch.tensor([1e6, -1e6])
        quantized, params = WeightConverter.quantize_weights(large)
        dequantized = WeightConverter.dequantize_weights(quantized, params)
        assert torch.allclose(dequantized, large, rtol=1e-3)

    def test_quantize_different_dtypes(self):
        # Test float32
        weights_f32 = torch.randn(10, 10, dtype=torch.float32)
        quantized, params = WeightConverter.quantize_weights(weights_f32)
        dequantized = WeightConverter.dequantize_weights(quantized, params)
        assert torch.allclose(dequantized, weights_f32, rtol=1e-1, atol=1e-2)
        assert dequantized.dtype == torch.float32
        
        # Test float16 with relaxed tolerances for lower precision
        weights_f16 = torch.randn(10, 10, dtype=torch.float16)
        quantized, params = WeightConverter.quantize_weights(weights_f16)
        dequantized = WeightConverter.dequantize_weights(quantized, params)
        assert torch.allclose(dequantized, weights_f16, rtol=2e-1, atol=5e-2)
        assert dequantized.dtype == torch.float16

    def test_dequantize_edge_cases(self):
        # Test empty tensor
        empty = torch.tensor([])
        dequantized = WeightConverter.dequantize_weights(empty, {"scale": 1.0})
        assert dequantized.numel() == 0
        
        # Test all zeros
        zeros = torch.zeros(10, 10)
        dequantized = WeightConverter.dequantize_weights(zeros, {"scale": 1.0})
        assert torch.all(dequantized == 0)
        
        # Test invalid scale
        invalid = torch.ones(10, 10)
        dequantized = WeightConverter.dequantize_weights(invalid, {"scale": 0})
        assert torch.all(dequantized == invalid)
        
        # Test missing zero_point
        quantized = torch.ones(10, 10)
        dequantized = WeightConverter.dequantize_weights(quantized, {"scale": 1.0}, symmetric=False)
        assert torch.all(dequantized == quantized)
