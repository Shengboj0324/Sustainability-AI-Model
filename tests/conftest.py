"""
Pytest configuration and fixtures
"""

import pytest
import torch
from PIL import Image
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing"""
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def sample_chat_messages():
    """Sample chat messages for testing"""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How do I recycle plastic bottles?"}
    ]


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return {
        "model": {
            "base_model_name": "test-model",
            "num_classes": 10
        },
        "data": {
            "input_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    }


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create temporary directory for model files"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture(scope="session")
def device():
    """Get available device for testing"""
    return "cuda" if torch.cuda.is_available() else "cpu"

