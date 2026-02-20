"""Test configuration and fixtures."""

import pytest
import torch
from pathlib import Path
from typing import Dict, Any

from src.adaptive_model_routing_serving_optimizer.utils.config import Config
from src.adaptive_model_routing_serving_optimizer.models.model import AdaptiveRoutingModel
from src.adaptive_model_routing_serving_optimizer.data.loader import ModelZooLoader, SyntheticDataLoader


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Provide test configuration."""
    return {
        "seed": 42,
        "torch_seed": 42,
        "numpy_seed": 42,
        "model": {
            "architectures": ["resnet50"],
            "compression_variants": [
                {
                    "type": "fp32",
                    "precision": "float32",
                    "memory_multiplier": 1.0,
                    "latency_multiplier": 1.0
                },
                {
                    "type": "fp16",
                    "precision": "float16",
                    "memory_multiplier": 0.5,
                    "latency_multiplier": 0.8
                }
            ]
        },
        "data": {
            "batch_size": 4,
            "validation_split": 0.2,
            "num_workers": 0,
            "image_size": 224,
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        },
        "training": {
            "epochs": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "patience": 5,
            "min_delta": 0.001,
            "gradient_clip_norm": 1.0,
            "bandit": {
                "algorithm": "ucb",
                "exploration_param": 0.1,
                "update_frequency": 10,
                "memory_size": 100,
                "context_dim": 16
            }
        },
        "routing": {
            "sla_constraints": {
                "p99_latency_ms": 50,
                "accuracy_threshold": 0.95,
                "memory_limit_gb": 8
            },
            "reward_weights": {
                "latency": 0.4,
                "accuracy": 0.3,
                "memory": 0.2,
                "cost": 0.1
            },
            "features": [
                "request_complexity",
                "historical_latency",
                "gpu_memory_usage",
                "queue_length",
                "model_load",
                "time_of_day"
            ]
        },
        "serving": {
            "max_batch_size": 32,
            "max_queue_size": 100,
            "timeout_ms": 1000
        },
        "monitoring": {
            "metrics_interval": 5,
            "log_level": "INFO",
            "export_metrics": False
        },
        "hardware": {
            "device": "cpu",
            "mixed_precision": False,
            "compile_model": False
        },
        "evaluation": {
            "test_split": 0.1,
            "metrics": [
                "accuracy",
                "latency_p99",
                "memory_usage"
            ],
            "benchmark": {
                "num_requests": 100,
                "concurrent_users": [1, 2],
                "request_patterns": ["uniform"]
            }
        }
    }


@pytest.fixture
def config_obj(test_config: Dict[str, Any]) -> Config:
    """Provide Config object."""
    return Config(test_config)


@pytest.fixture
def adaptive_model(test_config: Dict[str, Any]) -> AdaptiveRoutingModel:
    """Provide adaptive routing model for testing."""
    model = AdaptiveRoutingModel(test_config)
    return model


@pytest.fixture
def model_zoo_loader(test_config: Dict[str, Any]) -> ModelZooLoader:
    """Provide model zoo loader for testing."""
    return ModelZooLoader(test_config)


@pytest.fixture
def synthetic_data_loader(test_config: Dict[str, Any]) -> SyntheticDataLoader:
    """Provide synthetic data loader for testing."""
    return SyntheticDataLoader(test_config)


@pytest.fixture
def sample_context(test_config: Dict[str, Any]) -> torch.Tensor:
    """Provide sample context vector."""
    context_dim = test_config["training"]["bandit"]["context_dim"]
    return torch.randn(context_dim)


@pytest.fixture
def sample_batch_context(test_config: Dict[str, Any]) -> torch.Tensor:
    """Provide sample batch of context vectors."""
    batch_size = test_config["data"]["batch_size"]
    context_dim = test_config["training"]["bandit"]["context_dim"]
    return torch.randn(batch_size, context_dim)


@pytest.fixture
def device() -> torch.device:
    """Provide device for testing."""
    return torch.device("cpu")  # Use CPU for testing


@pytest.fixture
def temp_checkpoint_dir(tmp_path) -> Path:
    """Provide temporary directory for checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir