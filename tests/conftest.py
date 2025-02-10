import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from vishwamai.model import VishwamaiModel, VishwamaiConfig
from vishwamai.generate import VishwamaiGenerator, GenerationConfig

@pytest.fixture
def config():
    return VishwamaiConfig()

@pytest.fixture
def model(config):
    return VishwamaiModel(config)

@pytest.fixture
def generator(model):
    gen_config = GenerationConfig()
    return VishwamaiGenerator(model, gen_config)