
import pytest
import torch
import gc

@pytest.fixture
def setup_teardown():
    # Setup
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    yield
    
    # Teardown
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    gc.collect()