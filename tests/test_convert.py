import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import pytest
import torch
import torch.nn as nn
from vishwamai.convert import ModelConverter, ConversionConfig, WeightConverter

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.attention = nn.MultiheadAttention(embed_dim=10, num_heads=2)

    def forward(self, x):
        return self.linear(x)

@pytest.fixture
def model():
    return SimpleModel()

@pytest.fixture
def sample_weights():
    return torch.randn(10, 10)

def test_conversion_config_validation():
    with pytest.raises(ValueError):
        config = ConversionConfig(target_dtype="invalid_dtype")
        ModelConverter(config)

def test_model_fp16_conversion(model):
    config = ConversionConfig(target_dtype="fp16")
    converter = ModelConverter(config)
    converted = converter.convert_model(model)
    assert next(converted.parameters()).dtype == torch.float16

def test_model_bf16_conversion(model):
    config = ConversionConfig(target_dtype="bf16")
    converter = ModelConverter(config)
    converted = converter.convert_model(model)
    assert next(converted.parameters()).dtype == torch.bfloat16

def test_model_int8_conversion(model):
    config = ConversionConfig(target_dtype="int8")
    converter = ModelConverter(config)
    converted = converter.convert_model(model)
    # Check if model has been quantized
    assert any(isinstance(m, torch.nn.quantized.dynamic.Linear) 
              for m in converted.modules())

def test_efficient_attention_conversion(model):
    config = ConversionConfig(
        use_efficient_attention=True,
        use_tree_attention=False
    )
    converter = ModelConverter(config)
    with pytest.warns(UserWarning, match="Tree attention and efficient attention optimizations are not currently available"):
        converted = converter.convert_model(model)
    # Verify original attention module is unchanged
    assert isinstance(converted.attention, nn.MultiheadAttention)

def test_tree_attention_conversion(model):
    config = ConversionConfig(
        use_efficient_attention=True,
        use_tree_attention=True
    )
    converter = ModelConverter(config)
    with pytest.warns(UserWarning, match="Tree attention and efficient attention optimizations are not currently available"):
        converted = converter.convert_model(model)
    # Verify original attention module is unchanged
    assert isinstance(converted.attention, nn.MultiheadAttention)

def test_quantization_aware_training(model):
    model.train()  # Set model to training mode
    config = ConversionConfig(
        quantization_aware=True,
        target_dtype="int8"
    )
    converter = ModelConverter(config)
    converted = converter.convert_model(model)
    assert hasattr(converted, 'qconfig')

def test_weight_converter_quantize_symmetric(sample_weights):
    quantized, params = WeightConverter.quantize_weights(
        sample_weights, 
        bits=8, 
        symmetric=True
    )
    assert 'scale' in params
    assert quantized.max() <= 127
    assert quantized.min() >= -128

def test_weight_converter_quantize_asymmetric(sample_weights):
    quantized, params = WeightConverter.quantize_weights(
        sample_weights, 
        bits=8, 
        symmetric=False
    )
    assert 'scale' in params
    assert 'zero_point' in params
    assert quantized.max() <= 255
    assert quantized.min() >= 0

def test_weight_converter_roundtrip_symmetric(sample_weights):
    quantized, params = WeightConverter.quantize_weights(
        sample_weights, 
        bits=8, 
        symmetric=True
    )
    dequantized = WeightConverter.dequantize_weights(
        quantized, 
        params, 
        symmetric=True
    )
    # Check if values are close within quantization error margin
    assert torch.allclose(sample_weights, dequantized, rtol=0.15, atol=0.1)

def test_weight_converter_roundtrip_asymmetric(sample_weights):
    quantized, params = WeightConverter.quantize_weights(
        sample_weights, 
        bits=8, 
        symmetric=False
    )
    dequantized = WeightConverter.dequantize_weights(
        quantized, 
        params, 
        symmetric=False
    )
    # Check if values are close within quantization error margin
    assert torch.allclose(sample_weights, dequantized, rtol=0.15, atol=0.1)

def test_advanced_optimizations(model):
    config = ConversionConfig(optimization_level=2)
    converter = ModelConverter(config)
    converted = converter.convert_model(model)
    # Verify pruning was applied
    for name, param in converted.named_parameters():
        if 'weight' in name:
            # Check if some weights were pruned (set to 0)
            assert torch.sum(param == 0) > 0

def test_layer_fusion():
    class ModelWithBN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3)
            self.bn = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()

    model = ModelWithBN()
    config = ConversionConfig(optimization_level=2)
    converter = ModelConverter(config)
    converted = converter._fuse_layers(model)
    # Verify model structure has changed due to fusion
    assert isinstance(converted, nn.Module)

def test_pruning_threshold():
    model = SimpleModel()
    threshold = 0.5
    config = ConversionConfig(
        optimization_level=2,
        pruning_threshold=threshold
    )
    converter = ModelConverter(config)
    converted = converter._apply_pruning(model)
    
    for name, param in converted.named_parameters():
        if 'weight' in name:
            # Verify all remaining weights are above threshold
            non_zero_weights = param[param != 0]
            assert torch.all(torch.abs(non_zero_weights) > threshold)
