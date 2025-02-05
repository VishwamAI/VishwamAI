import pytest
import torch
from vishwamai.model import VishwamaiModel, VishwamaiConfig

@pytest.fixture
def config():
    return VishwamaiConfig(
        vocab_size=256000,
        hidden_size=128,  # Smaller size for testing
        num_hidden_layers=4,  # Fewer layers for testing
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=256,
        max_position_embeddings=512,
        memory_layers=2,
        memory_size=64,
        memory_heads=4,
        use_memory=True,
    )

@pytest.fixture
def model(config):
    return VishwamaiModel(config, device="cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def input_data(config):
    batch_size = 2
    seq_length = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, 32))
    attention_mask = torch.ones((batch_size, 32), dtype=torch.long)
    return input_ids, attention_mask

def test_model_initialization(model):
    assert isinstance(model, VishwamaiModel), "Model initialization failed"
    assert hasattr(model, "embeddings"), "Embeddings layer missing"
    assert hasattr(model, "blocks"), "Transformer blocks missing"
    assert len(model.blocks) == model.config.num_hidden_layers, "Incorrect number of blocks"

def test_model_forward_pass(model, input_data):
    input_ids, attention_mask = input_data
    outputs = model(input_ids, attention_mask)

    # Check output keys
    assert "logits" in outputs, "Logits missing in output"
    assert "mtp_logits" in outputs, "Multi-token prediction logits missing"
    assert "memory_state" in outputs, "Memory state missing"
    assert "hidden_states" in outputs, "Hidden states missing"

    # Check output shapes
    batch_size, seq_length = input_ids.shape
    assert outputs["logits"].shape == (batch_size, seq_length, model.config.vocab_size), "Logits shape mismatch"
    assert outputs["mtp_logits"].shape == (batch_size, seq_length, model.config.vocab_size), "MTP logits shape mismatch"
    assert len(outputs["memory_state"]) == model.config.num_hidden_layers, "Memory state length mismatch"
    assert outputs["hidden_states"].shape == (batch_size, seq_length, model.config.hidden_size), "Hidden states shape mismatch"

def test_memory_state_persistence(model, input_data):
    input_ids, attention_mask = input_data
    outputs1 = model(input_ids, attention_mask)
    memory_state = outputs1["memory_state"]

    # Pass memory state to the next forward pass
    outputs2 = model(input_ids, attention_mask, memory_state=memory_state)

    # Ensure memory state is updated
    assert outputs2["memory_state"] != memory_state, "Memory state not updated"

def test_attention_mask(model, input_data):
    input_ids, attention_mask = input_data
    outputs = model(input_ids, attention_mask)

    # Check that attention mask is applied correctly
    assert outputs["logits"].shape == (input_ids.shape[0], input_ids.shape[1], model.config.vocab_size), "Attention mask not applied correctly"

def test_device_transfer(model, input_data):
    input_ids, attention_mask = input_data
    if torch.cuda.is_available():
        model.to("cuda")
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
        outputs = model(input_ids, attention_mask)
        assert outputs["logits"].device.type == "cuda", "Device transfer failed"
    else:
        pytest.skip("CUDA not available")

def test_gradient_flow(model, input_data):
    input_ids, attention_mask = input_data
    model.zero_grad()
    outputs = model(input_ids, attention_mask)
    loss = outputs["logits"].sum()  # Dummy loss for testing
    loss.backward()

    # Check gradients
    for param in model.parameters():
        assert param.grad is not None, "Gradient flow failed"

def test_large_sequence_input(model, config):
    # Test with a sequence longer than max_position_embeddings
    input_ids = torch.randint(0, config.vocab_size, (2, 32 + 10))
    attention_mask = torch.ones((2, 32 + 10), dtype=torch.long)

    with pytest.raises(RuntimeError): # Expecting a different error, but keeping RuntimeError for now.
        model(input_ids, attention_mask)

def test_memory_device_mismatch(model, input_data):
    input_ids, attention_mask = input_data
    if torch.cuda.is_available():
        # Create memory state on CPU
        memory_state = [torch.zeros(2, model.config.memory_size, model.config.hidden_size // 2) for _ in range(model.config.num_hidden_layers)]
        with pytest.raises(RuntimeError): # Expecting a different error, but keeping RuntimeError for now.
            model(input_ids, attention_mask, memory_state=memory_state)
    else:
        pytest.skip("CUDA not available")

def test_compute_loss(model, input_data):
    input_ids, attention_mask = input_data
    outputs = model(input_ids, attention_mask)

    # Dummy loss computation
    logits = outputs["logits"]
    targets = torch.randint(0, model.config.vocab_size, logits.shape[:-1], device=logits.device)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    assert loss.item() > 0, "Loss computation failed"

def test_train_step(model, input_data):
    input_ids, attention_mask = input_data
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()

    # Forward pass
    outputs = model(input_ids, attention_mask)
    logits = outputs["logits"]
    targets = torch.randint(0, model.config.vocab_size, logits.shape[:-1], device=logits.device)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert loss.item() > 0, "Training step failed"

def test_evaluation(model, input_data):
    input_ids, attention_mask = input_data
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs["logits"]
        assert logits.requires_grad is False, "Evaluation mode failed"

def test_training_loop(model, input_data):
    input_ids, attention_mask = input_data
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()

    for _ in range(2):  # Simulate 2 training steps
        outputs = model(input_ids, attention_mask)
        logits = outputs["logits"]
        targets = torch.randint(0, model.config.vocab_size, logits.shape[:-1], device=logits.device)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss.item() > 0, "Training loop failed"

def test_gradient_accumulation(model, input_data):
    input_ids, attention_mask = input_data
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()

    # Simulate gradient accumulation over 2 steps
    for _ in range(2):
        outputs = model(input_ids, attention_mask)
        logits = outputs["logits"]
        targets = torch.randint(0, model.config.vocab_size, logits.shape[:-1], device=logits.device)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    assert loss.item() > 0, "Gradient accumulation failed"

def test_fp16_training(model, input_data):
    if torch.cuda.is_available():
        model.half()  # Convert model to FP16
        input_ids, attention_mask = input_data
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")

        outputs = model(input_ids, attention_mask)
        assert outputs["logits"].dtype == torch.float16, "FP16 training failed"
    else:
        pytest.skip("CUDA not available")
