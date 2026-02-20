"""Tests for data loading and preprocessing modules."""

import pytest
import torch
import numpy as np
from typing import Dict, Any

from src.adaptive_model_routing_serving_optimizer.data.loader import (
    ModelZooLoader,
    SyntheticDataLoader,
    create_data_loaders
)
from src.adaptive_model_routing_serving_optimizer.data.preprocessing import (
    RequestPreprocessor,
    ContextExtractor,
    SystemMonitor
)


class TestModelZooLoader:
    """Test cases for ModelZooLoader."""

    def test_initialization(self, model_zoo_loader: ModelZooLoader) -> None:
        """Test ModelZooLoader initialization."""
        assert len(model_zoo_loader.architectures) > 0
        assert len(model_zoo_loader.compression_variants) > 0
        assert hasattr(model_zoo_loader, 'model_registry')

    def test_load_model_fp32(self, model_zoo_loader: ModelZooLoader) -> None:
        """Test loading FP32 model."""
        model = model_zoo_loader.load_model("resnet50", "fp32")
        assert model is not None
        assert next(model.parameters()).dtype == torch.float32

    def test_load_model_fp16(self, model_zoo_loader: ModelZooLoader) -> None:
        """Test loading FP16 model."""
        model = model_zoo_loader.load_model("resnet50", "fp16")
        assert model is not None
        assert next(model.parameters()).dtype == torch.float16

    def test_unsupported_architecture(self, model_zoo_loader: ModelZooLoader) -> None:
        """Test error handling for unsupported architecture."""
        with pytest.raises(ValueError, match="Unsupported architecture"):
            model_zoo_loader.load_model("unsupported_model")

    def test_get_model_variants(self, model_zoo_loader: ModelZooLoader) -> None:
        """Test getting all variants of a model."""
        variants = model_zoo_loader.get_model_variants("resnet50")
        assert len(variants) > 0
        assert "fp32" in variants
        assert "fp16" in variants

    def test_compression_effects(self, model_zoo_loader: ModelZooLoader) -> None:
        """Test that compression actually changes the model."""
        fp32_model = model_zoo_loader.load_model("resnet50", "fp32")
        fp16_model = model_zoo_loader.load_model("resnet50", "fp16")

        fp32_param = next(fp32_model.parameters())
        fp16_param = next(fp16_model.parameters())

        assert fp32_param.dtype != fp16_param.dtype


class TestSyntheticDataLoader:
    """Test cases for SyntheticDataLoader."""

    def test_initialization(self, synthetic_data_loader: SyntheticDataLoader) -> None:
        """Test SyntheticDataLoader initialization."""
        assert synthetic_data_loader.image_size == 224
        assert synthetic_data_loader.batch_size == 4

    def test_generate_request_context(self, synthetic_data_loader: SyntheticDataLoader) -> None:
        """Test context generation."""
        context = synthetic_data_loader.generate_request_context(batch_size=5)

        assert isinstance(context, dict)
        assert len(context) > 0

        # Check all context tensors have correct batch size
        for key, tensor in context.items():
            assert tensor.shape[0] == 5
            assert tensor.shape[1] == 1  # Feature dimension

    def test_generate_image_batch(self, synthetic_data_loader: SyntheticDataLoader) -> None:
        """Test image batch generation."""
        batch_size = 8
        images = synthetic_data_loader.generate_image_batch(batch_size)

        assert images.shape == (batch_size, 3, 224, 224)
        assert images.dtype == torch.float32

        # Check normalization
        assert images.mean().abs() < 1.0  # Should be normalized

    def test_create_benchmark_dataset(self, synthetic_data_loader: SyntheticDataLoader) -> None:
        """Test benchmark dataset creation."""
        num_samples = 50
        dataset = synthetic_data_loader.create_benchmark_dataset(num_samples)

        assert len(dataset) == num_samples

        # Test dataset item
        image, context = dataset[0]
        assert image.shape == (3, 224, 224)
        assert isinstance(context, dict)

    def test_reproducibility(self, test_config: Dict[str, Any]) -> None:
        """Test that data generation is reproducible."""
        loader1 = SyntheticDataLoader(test_config)
        loader2 = SyntheticDataLoader(test_config)

        context1 = loader1.generate_request_context(batch_size=3)
        context2 = loader2.generate_request_context(batch_size=3)

        # Should be identical due to fixed seed
        for key in context1:
            assert torch.allclose(context1[key], context2[key], atol=1e-6)


class TestRequestPreprocessor:
    """Test cases for RequestPreprocessor."""

    def test_initialization(self, test_config: Dict[str, Any]) -> None:
        """Test RequestPreprocessor initialization."""
        preprocessor = RequestPreprocessor(test_config)
        assert preprocessor.context_dim == 16
        assert len(preprocessor.features) > 0

    def test_extract_features(self, test_config: Dict[str, Any]) -> None:
        """Test feature extraction."""
        preprocessor = RequestPreprocessor(test_config)

        request_data = {
            "batch_size": 4,
            "image_size": 224,
            "priority": 3,
            "accuracy_requirement": 0.95,
            "queue_length": 10
        }

        features = preprocessor.extract_features(request_data)

        assert features.shape == (16,)  # context_dim
        assert features.dtype == torch.float32

    def test_update_history(self, test_config: Dict[str, Any]) -> None:
        """Test history update functionality."""
        preprocessor = RequestPreprocessor(test_config)

        # Initially empty
        assert len(preprocessor.latency_history) == 0

        # Add some history
        preprocessor.update_history(45.0, 0.6, 5)
        preprocessor.update_history(50.0, 0.7, 8)

        assert len(preprocessor.latency_history) == 2
        assert len(preprocessor.memory_history) == 2
        assert len(preprocessor.queue_history) == 2

    def test_fit_normalizers(self, test_config: Dict[str, Any]) -> None:
        """Test feature normalizer fitting."""
        preprocessor = RequestPreprocessor(test_config)

        # Generate random features
        features = torch.randn(100, 16)
        preprocessor.fit_normalizers(features)

        # Check that mean and std are computed
        assert preprocessor.feature_stats["mean"].shape == (16,)
        assert preprocessor.feature_stats["std"].shape == (16,)

        # Check that normalization is approximately correct
        assert torch.allclose(preprocessor.feature_stats["mean"], features.mean(dim=0), atol=1e-5)
        assert torch.allclose(preprocessor.feature_stats["std"], features.std(dim=0), atol=1e-5)


class TestContextExtractor:
    """Test cases for ContextExtractor."""

    def test_initialization(self, test_config: Dict[str, Any]) -> None:
        """Test ContextExtractor initialization."""
        extractor = ContextExtractor(test_config)
        assert extractor.context_dim == 16
        assert hasattr(extractor, 'system_monitor')

    def test_extract_context(self, test_config: Dict[str, Any]) -> None:
        """Test context vector extraction."""
        extractor = ContextExtractor(test_config)

        request_data = {
            "batch_size": 2,
            "image_size": 224,
            "priority": 4,
            "accuracy_requirement": 0.98
        }

        context = extractor.extract_context(request_data)

        assert context.shape == (16,)
        assert context.dtype == torch.float32
        assert torch.all(torch.isfinite(context))

    def test_extract_batch_context(self, test_config: Dict[str, Any]) -> None:
        """Test batch context extraction."""
        extractor = ContextExtractor(test_config)

        batch_requests = [
            {"batch_size": 1, "priority": 3},
            {"batch_size": 2, "priority": 4},
            {"batch_size": 1, "priority": 2}
        ]

        contexts = extractor.extract_batch_context(batch_requests)

        assert contexts.shape == (3, 16)
        assert contexts.dtype == torch.float32

    def test_context_consistency(self, test_config: Dict[str, Any]) -> None:
        """Test that same input produces same context."""
        extractor = ContextExtractor(test_config)

        request_data = {"batch_size": 1, "priority": 3}

        context1 = extractor.extract_context(request_data)
        context2 = extractor.extract_context(request_data)

        # Should be very similar (allowing for small time differences)
        assert torch.allclose(context1, context2, atol=0.1)


class TestSystemMonitor:
    """Test cases for SystemMonitor."""

    def test_initialization(self, test_config: Dict[str, Any]) -> None:
        """Test SystemMonitor initialization."""
        monitor = SystemMonitor(test_config)
        assert monitor.config == test_config

    def test_get_current_state(self, test_config: Dict[str, Any]) -> None:
        """Test current state retrieval."""
        monitor = SystemMonitor(test_config)
        state = monitor.get_current_state()

        assert isinstance(state, dict)
        assert "gpu_memory_usage" in state
        assert "cpu_usage" in state
        assert "memory_usage" in state

        # Check value ranges
        assert 0 <= state["gpu_memory_usage"] <= 1
        assert 0 <= state["cpu_usage"] <= 1
        assert 0 <= state["memory_usage"] <= 1

    def test_get_performance_metrics(self, test_config: Dict[str, Any]) -> None:
        """Test performance metrics retrieval."""
        monitor = SystemMonitor(test_config)
        metrics = monitor.get_performance_metrics()

        assert isinstance(metrics, dict)
        assert "latency_p99" in metrics
        assert "throughput_rps" in metrics
        assert "error_rate" in metrics

        # Check reasonable value ranges
        assert metrics["latency_p99"] > 0
        assert metrics["throughput_rps"] > 0
        assert 0 <= metrics["error_rate"] <= 1


class TestDataLoaders:
    """Test cases for data loader creation."""

    def test_create_data_loaders(self, test_config: Dict[str, Any]) -> None:
        """Test data loader creation."""
        train_loader, val_loader = create_data_loaders(test_config)

        assert train_loader is not None
        assert val_loader is not None

        # Test loading a batch
        train_batch = next(iter(train_loader))
        images, contexts = train_batch

        assert images.shape[0] == test_config["data"]["batch_size"]
        assert images.shape[1:] == (3, 224, 224)
        assert isinstance(contexts, dict)

    def test_data_loader_consistency(self, test_config: Dict[str, Any]) -> None:
        """Test that data loaders produce consistent batches."""
        train_loader1, _ = create_data_loaders(test_config)
        train_loader2, _ = create_data_loaders(test_config)

        batch1 = next(iter(train_loader1))
        batch2 = next(iter(train_loader2))

        # Should be identical due to fixed seeds
        images1, contexts1 = batch1
        images2, contexts2 = batch2

        assert torch.allclose(images1, images2, atol=1e-6)

        # Check contexts are similar (allowing for time differences)
        for key in contexts1:
            if "timestamp" not in key and "time" not in key:
                assert torch.allclose(contexts1[key], contexts2[key], atol=1e-6)