#!/usr/bin/env python3
"""
Comprehensive VishwamAI demonstration script.

This script showcases the key features of VishwamAI including:
- Model architecture and configuration
- Hardware optimization detection
- Memory usage estimation
- Basic training simulation
- Text generation
- Multimodal processing
"""

import jax
import jax.numpy as jnp
import time
from pathlib import Path

# Import VishwamAI components
from vishwamai import (
    ModelConfig, VishwamAIModel, TrainingConfig, CurriculumTrainer,
    MultimodalProcessor, TextGenerator, pipeline,
    get_hardware_info, estimate_memory_usage, print_model_info,
    setup_mixed_precision, count_parameters, format_number
)


def demonstrate_hardware_detection():
    """Demonstrate hardware detection and optimization."""
    print("🖥️  Hardware Detection")
    print("-" * 30)
    
    hw_info = get_hardware_info()
    print(f"Available devices: {hw_info['num_devices']}")
    print(f"Device types: {set(hw_info['device_types'])}")
    print(f"Has TPU: {hw_info['has_tpu']}")
    print(f"Has GPU: {hw_info['has_gpu']}")
    
    # Mixed precision setup
    mp_config = setup_mixed_precision()
    print(f"Mixed precision: {mp_config['use_mixed_precision']}")
    print(f"Compute dtype: {mp_config['dtype']}")
    
    return hw_info, mp_config


def demonstrate_model_configurations():
    """Demonstrate different model configurations."""
    print("\n🏗️  Model Configurations")
    print("-" * 30)
    
    configs = {
        "Tiny": ModelConfig(
            dim=512, depth=6, heads=8, vocab_size=10000, max_seq_len=512
        ),
        "Small": ModelConfig(
            dim=1024, depth=12, heads=16, vocab_size=32000, max_seq_len=2048
        ),
        "Medium": ModelConfig(
            dim=2048, depth=24, heads=32, vocab_size=50304, max_seq_len=4096,
            enable_multimodal=True
        ),
    }
    
    for name, config in configs.items():
        memory = estimate_memory_usage(config, batch_size=8)
        print(f"\n{name} Model:")
        print(f"  Parameters: {format_number(memory['parameter_count'])}")
        print(f"  Inference: {memory['inference_gb']:.2f} GB")
        print(f"  Training: {memory['total_gb']:.2f} GB")
    
    return configs["Small"]  # Return small config for further demos


def demonstrate_model_creation(config: ModelConfig):
    """Demonstrate model creation and initialization."""
    print("\n🤖 Model Creation")
    print("-" * 30)
    
    # Create model
    model = VishwamAIModel(config)
    print(f"✅ Model created with {config.depth} layers")
    
    # Initialize parameters
    rng_key = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((1, 64), dtype=jnp.int32)
    
    print("Initializing parameters...")
    start_time = time.time()
    params = model.init(rng_key, dummy_input, training=False)
    init_time = time.time() - start_time
    
    param_count = count_parameters(params)
    print(f"✅ Initialized {format_number(param_count)} parameters in {init_time:.2f}s")
    
    return model, params


def demonstrate_forward_pass(model: VishwamAIModel, params, config: ModelConfig):
    """Demonstrate model forward pass."""
    print("\n⚡ Forward Pass")
    print("-" * 30)
    
    # Create test input
    batch_size, seq_len = 2, 128
    test_input = jax.random.randint(
        jax.random.PRNGKey(123),
        (batch_size, seq_len),
        minval=0,
        maxval=config.vocab_size
    )
    
    print(f"Input shape: {test_input.shape}")
    
    # Compile forward function
    @jax.jit
    def forward_fn(params, x):
        return model.apply(params, x, training=False)
    
    # Warmup
    _ = forward_fn(params, test_input)
    
    # Benchmark
    times = []
    for _ in range(5):
        start_time = time.time()
        output = forward_fn(params, test_input)
        output.block_until_ready()
        times.append(time.time() - start_time)
    
    avg_time = sum(times) / len(times)
    print(f"Output shape: {output.shape}")
    print(f"Forward pass time: {avg_time*1000:.2f}ms")
    print(f"Throughput: {batch_size/avg_time:.1f} samples/sec")
    
    return output


def demonstrate_text_generation(model: VishwamAIModel, params, config: ModelConfig):
    """Demonstrate text generation."""
    print("\n📝 Text Generation")
    print("-" * 30)
    
    # Create text generator
    generator = TextGenerator(model, params, config=config)
    
    # Simple token generation (without real tokenizer)
    print("Generating sequence...")
    
    # Start with a few tokens
    prompt_tokens = [1, 100, 200, 300]  # Dummy tokens
    
    try:
        generated = generator.generate(
            prompt_tokens,
            max_length=20,
            temperature=0.8,
            seed=42
        )
        print(f"Input tokens: {prompt_tokens}")
        print(f"Generated tokens: {generated}")
        print("✅ Text generation successful")
        
    except Exception as e:
        print(f"⚠️  Text generation demo limited (no tokenizer): {e}")


def demonstrate_multimodal_processing(config: ModelConfig):
    """Demonstrate multimodal processing."""
    print("\n🎭 Multimodal Processing")
    print("-" * 30)
    
    if not config.enable_multimodal:
        print("⚠️  Multimodal not enabled in this config")
        return
    
    # Create multimodal processor
    processor = MultimodalProcessor(
        vocab_size=config.vocab_size,
        embed_dim=config.dim,
        vision_config={"image_size": 224, "patch_size": 16},
        audio_config={"n_mels": 80}
    )
    
    # Dummy multimodal inputs
    text_ids = jnp.array([[1, 100, 200, 300, 2]], dtype=jnp.int32)  # [batch, seq]
    images = jnp.ones((1, 224, 224, 3), dtype=jnp.float32)  # [batch, h, w, c]
    audio = jnp.ones((1, 100, 80), dtype=jnp.float32)  # [batch, time, mels]
    
    print(f"Text shape: {text_ids.shape}")
    print(f"Image shape: {images.shape}")
    print(f"Audio shape: {audio.shape}")
    
    # Initialize processor
    rng_key = jax.random.PRNGKey(456)
    processor_params = processor.init(rng_key, text_ids, images, audio)
    
    # Process multimodal input
    embeddings = processor.apply(
        processor_params,
        text_ids=text_ids,
        images=images,
        audio=audio,
        training=False
    )
    
    print(f"Combined embeddings shape: {embeddings.shape}")
    print("✅ Multimodal processing successful")


def demonstrate_training_setup(config: ModelConfig):
    """Demonstrate training configuration setup."""
    print("\n🏋️  Training Setup")
    print("-" * 30)
    
    # Create training configuration
    training_config = TrainingConfig(
        model_config=config,
        batch_size=4,
        learning_rate=1e-4,
        total_steps=1000,
        use_curriculum=True,
        use_lora=True,  # Parameter-efficient training
        gradient_checkpointing=True
    )
    
    print(f"Batch size: {training_config.batch_size}")
    print(f"Learning rate: {training_config.learning_rate}")
    print(f"Total steps: {training_config.total_steps}")
    print(f"Curriculum learning: {training_config.use_curriculum}")
    print(f"LoRA enabled: {training_config.use_lora}")
    
    # Create trainer (but don't actually train)
    trainer = CurriculumTrainer(training_config)
    
    # Show curriculum stages
    print("\nCurriculum stages:")
    for i, stage in enumerate(training_config.curriculum_stages):
        print(f"  {i+1}. {stage['name']}: {stage['steps']} steps, {stage['max_seq_len']} tokens")
    
    print("✅ Training setup complete")


def demonstrate_pipeline_usage():
    """Demonstrate high-level pipeline usage."""
    print("\n🚀 Pipeline Usage")
    print("-" * 30)
    
    try:
        # Create text generation pipeline
        generator = pipeline("text-generation")
        print("✅ Text generation pipeline created")
        
        # Create multimodal pipeline
        multimodal_gen = pipeline("multimodal-generation")
        print("✅ Multimodal generation pipeline created")
        
        print("Pipelines ready for inference!")
        
    except Exception as e:
        print(f"⚠️  Pipeline demo limited: {e}")


def main():
    """Run comprehensive VishwamAI demonstration."""
    
    print("🚀 VishwamAI Comprehensive Demonstration")
    print("=" * 50)
    
    try:
        # Hardware detection
        hw_info, mp_config = demonstrate_hardware_detection()
        
        # Model configurations
        demo_config = demonstrate_model_configurations()
        
        # Model creation
        model, params = demonstrate_model_creation(demo_config)
        
        # Forward pass
        output = demonstrate_forward_pass(model, params, demo_config)
        
        # Text generation
        demonstrate_text_generation(model, params, demo_config)
        
        # Multimodal processing
        demonstrate_multimodal_processing(demo_config)
        
        # Training setup
        demonstrate_training_setup(demo_config)
        
        # Pipeline usage
        demonstrate_pipeline_usage()
        
        print("\n🎉 Demonstration completed successfully!")
        print("\nNext steps:")
        print("• Try training: python scripts/train_vishwamai.py --config configs/small_model.json")
        print("• Try inference: python scripts/inference.py --prompt 'Hello, VishwamAI!'")
        print("• Explore configs: ls configs/")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nTroubleshooting:")
        print("• Check dependencies: python setup_vishwamai.py")
        print("• Verify hardware: jax.devices()")
        print("• Check memory: reduce batch size or model size")


if __name__ == '__main__':
    main()
