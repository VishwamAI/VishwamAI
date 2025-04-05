"""Test suite for VishwamAI pretraining components."""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'  # Force CPU execution

import pytest
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Dict, Any
import optax
from jax import lax
from flax.training import dynamic_scale

from vishwamai.model import VishwamAI, VishwamAIConfig
from vishwamai.transformer import EnhancedTransformerModel, TPUTrainingState
from vishwamai.pipeline import TPUDataPipeline, DistillationDataPipeline
from vishwamai.device_mesh import TPUMeshContext
from vishwamai.distill import (
    create_student_model,
    initialize_from_teacher,
    DistillationTrainer
)
from vishwamai.thoughts import TreeOfThoughts
from vishwamai.configs.tpu_v3_config import TPUV3Config
from vishwamai.configs.budget_model_config import BudgetModelConfig

DynamicScale = dynamic_scale.DynamicScale

@pytest.fixture
def training_config():
    """Provide test training configuration."""
    return {
        "model": {
            "vocab_size": 1000,
            "hidden_dim": 128,
            "num_layers": 2,
            "num_heads": 4,
            "head_dim": 32,
            "mlp_dim": 512,
            "max_seq_len": 64,
            "dropout_rate": 0.1,
            "attention_dropout": 0.1,
            "use_flash_attn": False
        },
        "training": {
            "batch_size": 4,
            "grad_accum_steps": 1,
            "learning_rate": 1e-4,
            "warmup_steps": 10,
            "max_steps": 20,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0
        },
        "thinking": {
            "num_steps": 2,
            "max_branches": 2,
            "max_depth": 2,
            "beam_width": 2,
            "temperature": 0.7
        },
        "distillation": {
            "teacher_model": "test-teacher",
            "temperature": 2.0,
            "alpha": 0.5,
            "layer_mapping_strategy": "uniform"
        }
    }

@pytest.fixture
def student_model(training_config):
    """Create a test student model."""
    config = VishwamAIConfig(**training_config["model"])
    return VishwamAI(config=config)

@pytest.fixture
def teacher_model(training_config):
    """Create a test teacher model."""
    config = VishwamAIConfig(**training_config["model"])
    return VishwamAI(config=config)

@pytest.fixture
def dummy_input(training_config):
    """Provide dummy input for testing."""
    return jnp.ones((4, training_config["model"]["max_seq_len"]), dtype=jnp.int32)

@pytest.fixture
def rng_key():
    """Provide random key for initialization."""
    return jax.random.PRNGKey(0)

def test_config_loading():
    """Test configuration loading and validation."""
    config = TPUV3Config()
    assert isinstance(config.model_config, dict)
    assert isinstance(config.training_config, dict)
    assert isinstance(config.tpu_config, dict)
    assert isinstance(config.memory_config, dict)

def test_model_creation(training_config):
    """Test model creation with config."""
    config = VishwamAIConfig(**training_config["model"])
    model = VishwamAI(config=config)
    assert isinstance(model, VishwamAI)

def test_distillation_setup(student_model, teacher_model, training_config):
    """Test distillation training setup."""
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_config=training_config["model"],
        temperature=training_config["distillation"]["temperature"],
        alpha=training_config["distillation"]["alpha"],
        use_flash_attn=training_config["model"]["use_flash_attn"]
    )
    assert isinstance(trainer, DistillationTrainer)

def test_tree_of_thoughts_setup(student_model):
    """Test Tree of Thoughts initialization."""
    tot = TreeOfThoughts(
        model=student_model,
        params=None,
        tokenizer=None,
        max_branches=2,
        max_depth=2,
        beam_width=2,
        temperature=0.7
    )
    assert isinstance(tot, TreeOfThoughts)

def test_tpu_mesh_setup():
    """Test TPU mesh initialization."""
    jax.config.update('jax_platform_name', 'cpu')
    devices = jax.devices()
    mesh_context = TPUMeshContext({"tpu": {"mesh_shape": [1]}}, data_parallel=True)
    assert mesh_context is not None

def test_data_pipeline_setup(training_config):
    """Test data pipeline initialization."""
    devices = jax.devices()
    pipeline = DistillationDataPipeline(
        config=training_config,
        teacher_model=None,
        devices=devices
    )
    assert isinstance(pipeline, DistillationDataPipeline)

def test_training_state_initialization(student_model):
    """Test training state setup."""
    state = TPUTrainingState(
        step=0,
        params=None,
        opt_state=None,
        model_fn=student_model.apply,
        tx=None
    )
    assert isinstance(state, TPUTrainingState)

def test_performance_profiling(training_config):
    """Test performance profiling setup."""
    from vishwamai.profiler import TPUProfiler
    profiler = TPUProfiler(training_config)
    assert hasattr(profiler, 'get_metrics_summary')

def test_gemm_performance(student_model, dummy_input, rng_key):
    """Test matrix multiplication kernel performance."""
    variables = student_model.init(rng_key, dummy_input)
    logits = student_model.apply(variables, dummy_input)
    assert logits is not None

def test_fp8_conversion():
    """Test FP8 data conversion utilities."""
    try:
        from vishwamai.kernels.cuda.fp8_cast_bf16 import convert_grads_fp8
        grads = {"weight": jnp.ones((64, 64), dtype=jnp.float32)}
        scales = {"weight": jnp.ones((64,), dtype=jnp.float32)}
        config = {"fp8_format": "e4m3"}
        converted = convert_grads_fp8(grads, scales, config)
        assert isinstance(converted, dict)
        assert "weight" in converted
    except ImportError:
        pytest.skip("fp8_mla_cuda not installed - skipping FP8 tests")

def test_flash_attention(student_model, training_config):
    """Test flash attention implementation."""
    batch_size = 2
    seq_len = 32
    num_heads = training_config["model"]["num_heads"]
    head_dim = training_config["model"]["head_dim"]
    shape = (batch_size, seq_len, num_heads, head_dim)
    query = jnp.ones(shape)
    key = jnp.ones(shape)
    value = jnp.ones(shape)
    output = student_model.attention(query, key, value)
    assert output.shape == shape

def test_model_parallel_setup():
    """Test model parallel initialization."""
    devices = jax.devices()
    mesh = jax.sharding.Mesh(devices, ('data',))
    assert mesh.devices.shape[0] > 0

def test_gradient_checkpointing(student_model, dummy_input, rng_key):
    """Test gradient checkpointing functionality."""
    from jax.experimental import checkify
    def forward_fn(params, x):
        return student_model.apply({"params": params}, x)
    checkpointed_fn = checkify.checkify(forward_fn)  # Use checkify for gradient checkpointing
    variables = student_model.init(rng_key, dummy_input)
    output = checkpointed_fn(variables["params"], dummy_input)
    assert output is not None  # Just check the output exists

def test_distillation_pipeline_batch_processing(training_config, teacher_model):
    """Test distillation pipeline batch processing."""
    batch = {
        "input_ids": jnp.ones((4, 32), dtype=jnp.int32),
        "labels": jnp.ones((4, 32), dtype=jnp.int32)
    }
    devices = jax.devices()
    pipeline = DistillationDataPipeline(
        config=training_config,
        teacher_model=teacher_model,
        devices=devices
    )
    processed_batch = pipeline.process_batch(batch)
    assert "teacher_logits" in processed_batch
    assert processed_batch["teacher_logits"].shape == (4, 32, training_config["model"]["vocab_size"])

def test_teacher_student_attention_alignment(student_model, teacher_model, training_config):
    """Test alignment of teacher and student attention patterns."""
    batch_size = 2
    seq_len = 16
    hidden_dim = training_config["model"]["hidden_dim"]
    
    # Use integer input IDs instead of hidden states
    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    
    # Initialize models
    rng = jax.random.PRNGKey(0)
    teacher_variables = teacher_model.init(rng, input_ids)
    student_variables = student_model.init(rng, input_ids)
    
    # Get attention patterns
    teacher_outputs = teacher_model.apply(teacher_variables, input_ids)
    student_outputs = student_model.apply(student_variables, input_ids)
    
    # Compare output shapes
    assert teacher_outputs["logits"].shape[-1] == student_outputs["logits"].shape[-1]

def test_optimized_kv_cache(student_model):
    """Test optimized key-value cache implementation."""
    batch_size = 2
    max_length = 32
    num_heads = 4
    head_dim = 32
    kv_cache = student_model.initialize_kv_cache(
        batch_size=batch_size,
        max_length=max_length,
        num_heads=num_heads,
        head_dim=head_dim
    )
    assert "keys" in kv_cache
    assert "values" in kv_cache
    assert kv_cache["keys"].shape == (batch_size, max_length, num_heads, head_dim)

def test_memory_efficient_attention(student_model, training_config):
    """Test memory-efficient attention implementation."""
    batch_size = 2
    seq_len = 32
    num_heads = training_config["model"]["num_heads"]
    head_dim = training_config["model"]["head_dim"]
    query = jnp.ones((batch_size, seq_len, num_heads, head_dim))
    key = jnp.ones((batch_size, seq_len, num_heads, head_dim))
    value = jnp.ones((batch_size, seq_len, num_heads, head_dim))
    initial_mem = jax.lib.xla_bridge.get_backend().live_buffers()
    output = student_model.memory_efficient_attention(query, key, value)
    final_mem = jax.lib.xla_bridge.get_backend().live_buffers()
    assert output.shape == (batch_size, seq_len, num_heads, head_dim)
    assert len(final_mem) - len(initial_mem) < query.size * query.dtype.itemsize

def test_mixed_precision_training(student_model, dummy_input, rng_key):
    """Test mixed precision training setup."""
    optimizer = optax.adam(learning_rate=1e-4)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optimizer)
    variables = student_model.init(rng_key, dummy_input)
    state = TPUTrainingState.create(
        apply_fn=student_model.apply,
        params=variables["params"],
        tx=optimizer,
        dynamic_scale=DynamicScale()
    )
    assert state.dynamic_scale is not None
    def train_step(state, batch):
        def loss_fn(params):
            outputs = state.apply_fn({"params": params}, batch)
            # Extract logits from outputs dict
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            return jnp.mean(logits)
        dynamic_scale = state.dynamic_scale
        grad_fn = dynamic_scale.value_and_grad(loss_fn)
        dynamic_scale, is_finite, aux, grads = grad_fn(state.params)
        assert is_finite
        return state.apply_gradients(grads=grads), aux
    state, loss = train_step(state, dummy_input)
    assert loss is not None

def test_full_pretraining_pipeline(training_config, student_model, teacher_model, dummy_input, rng_key):
    """Test full pretraining pipeline integration."""
    devices = jax.devices()
    mesh_context = TPUMeshContext({"tpu": {"mesh_shape": [1]}}, data_parallel=True)
    pipeline = DistillationDataPipeline(
        config=training_config,
        teacher_model=teacher_model,
        devices=devices,
        enable_thinking=True
    )
    dummy_dataset = {
        "input_ids": jnp.ones((10, 32), dtype=jnp.int32),
        "labels": jnp.ones((10, 32), dtype=jnp.int32)
    }
    optimizer = optax.adamw(
        learning_rate=training_config["training"]["learning_rate"],
        weight_decay=training_config["training"]["weight_decay"]
    )
    state = TPUTrainingState.create(
        apply_fn=student_model.apply,
        params=student_model.init(rng_key, dummy_input),
        tx=optimizer
    )
    with mesh_context:
        for _ in range(2):
            batch = jax.tree_util.tree_map(
                lambda x: x[:training_config["training"]["batch_size"]], 
                dummy_dataset
            )
            state, metrics = pipeline.train_step(state, batch)
    assert "loss" in metrics
    assert "learning_rate" in metrics

def test_tree_of_thoughts_reasoning(student_model, training_config):
    """Test Tree of Thoughts reasoning capabilities."""
    tot = TreeOfThoughts(
        model=student_model,
        params=None,
        tokenizer=None,
        max_branches=training_config["thinking"]["max_branches"],
        max_depth=training_config["thinking"]["max_depth"],
        beam_width=training_config["thinking"]["beam_width"],
        temperature=training_config["thinking"]["temperature"]
    )
    initial_prompt = "Solve: 2 + 2 = ?"
    thoughts = tot.generate_thoughts(initial_prompt=initial_prompt, num_samples=2)
    assert len(thoughts) > 0
    assert all(isinstance(t, str) for t in thoughts)
    scores = tot.evaluate_thoughts(thoughts)
    assert len(scores) == len(thoughts)
    assert all(isinstance(s, float) for s in scores)

def test_checkpoint_handling(student_model, training_config, dummy_input, rng_key, tmp_path):
    """Test model checkpoint saving and loading."""
    import os
    import pickle
    import numpy as np
    variables = student_model.init(rng_key, dummy_input)
    
    # Convert JAX arrays to numpy for serialization
    def convert_to_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_to_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(convert_to_numpy(x) for x in obj)
        elif hasattr(obj, 'numpy'):  # JAX arrays have numpy() method
            return np.array(obj)
        return obj
    
    checkpoint_path = os.path.join(tmp_path, "checkpoint.pkl")
    state_dict = {
        "model": convert_to_numpy(variables),
        "config": training_config,
        "step": 0
    }
    
    # Save checkpoint
    with open(checkpoint_path, "wb") as f:
        pickle.dump(state_dict, f)
        
    # Load checkpoint
    with open(checkpoint_path, "rb") as f:
        loaded_state = pickle.load(f)
        
    assert "model" in loaded_state
    assert "config" in loaded_state
    assert loaded_state["step"] == 0

def test_error_handling():
    """Test error handling in pretraining pipeline."""
    with pytest.raises(ValueError, match="vocab_size must be positive"):
        VishwamAIConfig(
            vocab_size=-1,
            hidden_dim=128,
            num_layers=2,
            num_heads=4
        )
    with pytest.raises(ValueError, match="mesh_shape must be positive"):
        TPUMeshContext(
            {"tpu": {"mesh_shape": [-1]}},
            data_parallel=True
        )
    with pytest.raises(ValueError, match="batch_size must be positive"):
        DistillationDataPipeline(
            config={"training": {"batch_size": 0}},
            teacher_model=None,
            devices=jax.devices()
        )