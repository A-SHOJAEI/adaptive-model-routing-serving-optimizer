"""Tests for training modules."""

import pytest
import torch
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, MagicMock

from src.adaptive_model_routing_serving_optimizer.training.trainer import (
    RoutingTrainer,
    OnlineTrainer
)
from src.adaptive_model_routing_serving_optimizer.data.loader import create_data_loaders


class TestRoutingTrainer:
    """Test cases for RoutingTrainer."""

    def test_initialization(
        self,
        adaptive_model,
        test_config: Dict[str, Any],
        device: torch.device
    ) -> None:
        """Test RoutingTrainer initialization."""
        trainer = RoutingTrainer(adaptive_model, test_config, device)

        assert trainer.model == adaptive_model
        assert trainer.config == test_config
        assert trainer.device == device
        assert trainer.epochs == test_config["training"]["epochs"]
        assert trainer.learning_rate == test_config["training"]["learning_rate"]

    def test_optimizer_initialization(
        self,
        adaptive_model,
        test_config: Dict[str, Any],
        device: torch.device
    ) -> None:
        """Test optimizer initialization."""
        trainer = RoutingTrainer(adaptive_model, test_config, device)

        assert trainer.bandit_optimizer is not None
        assert trainer.policy_optimizer is not None
        assert trainer.bandit_scheduler is not None
        assert trainer.policy_scheduler is not None

    @patch('src.adaptive_model_routing_serving_optimizer.training.trainer.mlflow')
    def test_mlflow_setup_disabled(
        self,
        mock_mlflow,
        adaptive_model,
        test_config: Dict[str, Any],
        device: torch.device
    ) -> None:
        """Test MLflow setup when disabled."""
        # Remove MLflow config
        config_no_mlflow = test_config.copy()
        if "mlflow" in config_no_mlflow:
            del config_no_mlflow["mlflow"]

        trainer = RoutingTrainer(adaptive_model, config_no_mlflow, device)

        assert not trainer.use_mlflow
        mock_mlflow.set_tracking_uri.assert_not_called()

    def test_process_context_batch(
        self,
        adaptive_model,
        test_config: Dict[str, Any],
        synthetic_data_loader,
        device: torch.device
    ) -> None:
        """Test context batch processing."""
        trainer = RoutingTrainer(adaptive_model, test_config, device)

        # Generate sample context batch
        context_batch = synthetic_data_loader.generate_request_context(batch_size=4)

        processed_contexts = trainer._process_context_batch(context_batch)

        assert processed_contexts.shape == (4, test_config["training"]["bandit"]["context_dim"])
        assert processed_contexts.device == device

    def test_save_and_load_checkpoint(
        self,
        adaptive_model,
        test_config: Dict[str, Any],
        device: torch.device,
        temp_checkpoint_dir: Path
    ) -> None:
        """Test checkpoint saving and loading."""
        trainer = RoutingTrainer(adaptive_model, test_config, device)

        # Save checkpoint
        checkpoint_path = temp_checkpoint_dir / "test_checkpoint.pth"
        metrics = {"train_loss": 0.5, "val_loss": 0.4}

        trainer._save_checkpoint(checkpoint_path, epoch=5, metrics=metrics)

        # Check file exists
        assert checkpoint_path.exists()

        # Load checkpoint
        loaded_data = trainer.load_checkpoint(str(checkpoint_path))

        assert loaded_data["epoch"] == 5
        assert loaded_data["metrics"] == metrics
        assert trainer.current_epoch == 5

    def test_training_step_integration(
        self,
        adaptive_model,
        test_config: Dict[str, Any],
        device: torch.device
    ) -> None:
        """Test integration of training step components."""
        trainer = RoutingTrainer(adaptive_model, test_config, device)

        # Create minimal data loaders
        train_loader, val_loader = create_data_loaders(test_config)

        # Take first batch
        train_batch = next(iter(train_loader))

        # Test that training step can run without errors
        trainer.model.train()

        try:
            # This should not raise any errors
            images, contexts = train_batch
            images = images.to(device)
            batch_contexts = trainer._process_context_batch(contexts)

            assert batch_contexts.shape[0] == images.shape[0]
            assert batch_contexts.device == device

        except Exception as e:
            pytest.fail(f"Training step failed with error: {e}")

    def test_validation_step_integration(
        self,
        adaptive_model,
        test_config: Dict[str, Any],
        device: torch.device
    ) -> None:
        """Test validation step integration."""
        trainer = RoutingTrainer(adaptive_model, test_config, device)

        # Create minimal data loaders
        train_loader, val_loader = create_data_loaders(test_config)

        # Take first batch
        val_batch = next(iter(val_loader))

        trainer.model.eval()

        try:
            images, contexts = val_batch
            images = images.to(device)
            batch_contexts = trainer._process_context_batch(contexts)

            # Should run without errors
            with torch.no_grad():
                logits, values = trainer.model.routing_policy(batch_contexts)
                assert logits.shape[0] == images.shape[0]

        except Exception as e:
            pytest.fail(f"Validation step failed with error: {e}")

    @patch('src.adaptive_model_routing_serving_optimizer.training.trainer.mlflow')
    def test_train_short_run(
        self,
        mock_mlflow,
        adaptive_model,
        test_config: Dict[str, Any],
        device: torch.device,
        temp_checkpoint_dir: Path
    ) -> None:
        """Test short training run."""
        # Reduce epochs for faster testing
        config_short = test_config.copy()
        config_short["training"]["epochs"] = 1

        trainer = RoutingTrainer(adaptive_model, config_short, device)

        # Create data loaders with small dataset
        train_loader, val_loader = create_data_loaders(config_short)

        # Limit to just a few batches
        train_batches = [next(iter(train_loader)) for _ in range(2)]
        val_batches = [next(iter(val_loader)) for _ in range(1)]

        class LimitedDataLoader:
            def __init__(self, batches):
                self.batches = batches

            def __iter__(self):
                return iter(self.batches)

            def __len__(self):
                return len(self.batches)

        limited_train_loader = LimitedDataLoader(train_batches)
        limited_val_loader = LimitedDataLoader(val_batches)

        # Run training
        history = trainer.train(limited_train_loader, limited_val_loader, str(temp_checkpoint_dir))

        assert "history" in history
        assert len(history["history"]) == 1  # One epoch

        # Check that checkpoints were created
        assert (temp_checkpoint_dir / "final_model.pth").exists()

    def test_early_stopping_logic(
        self,
        adaptive_model,
        test_config: Dict[str, Any],
        device: torch.device
    ) -> None:
        """Test early stopping logic."""
        # Set very small patience for testing
        config_early = test_config.copy()
        config_early["training"]["patience"] = 1
        config_early["training"]["min_delta"] = 1.0  # Large min_delta to trigger early stop

        trainer = RoutingTrainer(adaptive_model, config_early, device)

        # Simulate no improvement scenario
        trainer.best_val_loss = 0.5
        trainer.patience_counter = 0

        # Current validation loss is worse than best - min_delta
        current_val_loss = 0.6  # Worse than 0.5 - 1.0

        if current_val_loss < trainer.best_val_loss - trainer.min_delta:
            trainer.patience_counter = 0
        else:
            trainer.patience_counter += 1

        assert trainer.patience_counter == 1

        # Should trigger early stopping
        assert trainer.patience_counter >= trainer.patience

    def test_learning_rate_scheduling(
        self,
        adaptive_model,
        test_config: Dict[str, Any],
        device: torch.device
    ) -> None:
        """Test learning rate scheduling."""
        trainer = RoutingTrainer(adaptive_model, test_config, device)

        initial_bandit_lr = trainer.bandit_optimizer.param_groups[0]['lr']
        initial_policy_lr = trainer.policy_optimizer.param_groups[0]['lr']

        # Step schedulers
        trainer.bandit_scheduler.step(0.5)  # ReduceLROnPlateau
        trainer.policy_scheduler.step()     # CosineAnnealingLR

        # Learning rates should remain the same or change based on scheduler logic
        bandit_lr = trainer.bandit_optimizer.param_groups[0]['lr']
        policy_lr = trainer.policy_optimizer.param_groups[0]['lr']

        # At least one should be defined and finite
        assert isinstance(bandit_lr, float) and bandit_lr > 0
        assert isinstance(policy_lr, float) and policy_lr > 0


class TestOnlineTrainer:
    """Test cases for OnlineTrainer."""

    def test_initialization(
        self,
        adaptive_model,
        test_config: Dict[str, Any]
    ) -> None:
        """Test OnlineTrainer initialization."""
        online_trainer = OnlineTrainer(adaptive_model, test_config)

        assert online_trainer.model == adaptive_model
        assert online_trainer.config == test_config
        assert online_trainer.update_frequency == test_config["training"]["bandit"]["update_frequency"]
        assert online_trainer.total_requests == 0

    def test_process_request_no_feedback(
        self,
        adaptive_model,
        test_config: Dict[str, Any],
        sample_context: torch.Tensor
    ) -> None:
        """Test request processing without feedback."""
        online_trainer = OnlineTrainer(adaptive_model, test_config)

        variant = online_trainer.process_request(sample_context)

        assert isinstance(variant, int)
        assert 0 <= variant < adaptive_model.num_variants
        assert online_trainer.total_requests == 1
        assert len(online_trainer.performance_buffer) == 1

    def test_process_request_with_feedback(
        self,
        adaptive_model,
        test_config: Dict[str, Any],
        sample_context: torch.Tensor
    ) -> None:
        """Test request processing with feedback."""
        online_trainer = OnlineTrainer(adaptive_model, test_config)

        # First request without feedback
        variant1 = online_trainer.process_request(sample_context)

        # Second request with feedback from first
        performance_feedback = {
            "latency_ms": 45.0,
            "memory_mb": 800.0,
            "accuracy": 0.96
        }

        variant2 = online_trainer.process_request(sample_context, performance_feedback)

        assert isinstance(variant2, int)
        assert online_trainer.total_requests == 2
        assert len(online_trainer.performance_buffer) == 2

    def test_buffer_size_limit(
        self,
        adaptive_model,
        test_config: Dict[str, Any],
        sample_context: torch.Tensor
    ) -> None:
        """Test that performance buffer size is limited."""
        online_trainer = OnlineTrainer(adaptive_model, test_config)

        # Process many requests (more than buffer limit of 100)
        for i in range(150):
            online_trainer.process_request(sample_context)

        # Buffer should be limited to 100
        assert len(online_trainer.performance_buffer) == 100
        assert online_trainer.total_requests == 150

    def test_get_online_stats(
        self,
        adaptive_model,
        test_config: Dict[str, Any],
        sample_context: torch.Tensor
    ) -> None:
        """Test online statistics retrieval."""
        online_trainer = OnlineTrainer(adaptive_model, test_config)

        # Process some requests
        for _ in range(5):
            online_trainer.process_request(sample_context)

        stats = online_trainer.get_online_stats()

        assert isinstance(stats, dict)
        assert "total_requests" in stats
        assert "buffer_size" in stats
        assert "arm_counts" in stats

        assert stats["total_requests"] == 5
        assert stats["buffer_size"] == 5

    def test_feedback_integration(
        self,
        adaptive_model,
        test_config: Dict[str, Any],
        sample_context: torch.Tensor
    ) -> None:
        """Test feedback integration with model."""
        online_trainer = OnlineTrainer(adaptive_model, test_config)

        # Get initial model statistics
        initial_stats = adaptive_model.get_routing_statistics()
        initial_training_steps = initial_stats["training_steps"]

        # Process request without feedback
        variant = online_trainer.process_request(sample_context)

        # Process another request with feedback
        performance_feedback = {
            "latency_ms": 42.0,
            "accuracy": 0.97,
            "memory_mb": 750.0
        }

        online_trainer.process_request(sample_context, performance_feedback)

        # Model should have been updated
        updated_stats = adaptive_model.get_routing_statistics()
        assert updated_stats["training_steps"] > initial_training_steps

    def test_context_preservation(
        self,
        adaptive_model,
        test_config: Dict[str, Any],
        sample_context: torch.Tensor
    ) -> None:
        """Test that context is preserved correctly in buffer."""
        online_trainer = OnlineTrainer(adaptive_model, test_config)

        # Process request
        variant = online_trainer.process_request(sample_context)

        # Check buffer content
        assert len(online_trainer.performance_buffer) == 1
        stored_context, stored_variant = online_trainer.performance_buffer[0]

        assert torch.allclose(stored_context, sample_context)
        assert stored_variant == variant

    def test_multiple_requests_sequence(
        self,
        adaptive_model,
        test_config: Dict[str, Any]
    ) -> None:
        """Test sequence of multiple requests with feedback."""
        online_trainer = OnlineTrainer(adaptive_model, test_config)

        contexts = [torch.randn(test_config["training"]["bandit"]["context_dim"]) for _ in range(3)]
        performances = [
            {"latency_ms": 40.0, "accuracy": 0.96, "memory_mb": 800.0},
            {"latency_ms": 45.0, "accuracy": 0.95, "memory_mb": 850.0},
            {"latency_ms": 38.0, "accuracy": 0.97, "memory_mb": 750.0}
        ]

        variants = []

        # First request (no feedback)
        variant1 = online_trainer.process_request(contexts[0])
        variants.append(variant1)

        # Subsequent requests with feedback
        for i in range(1, 3):
            variant = online_trainer.process_request(contexts[i], performances[i-1])
            variants.append(variant)

        assert len(variants) == 3
        assert online_trainer.total_requests == 3
        assert all(0 <= v < adaptive_model.num_variants for v in variants)

    def test_online_stats_accuracy(
        self,
        adaptive_model,
        test_config: Dict[str, Any],
        sample_context: torch.Tensor
    ) -> None:
        """Test accuracy of online statistics."""
        online_trainer = OnlineTrainer(adaptive_model, test_config)

        # Process specific number of requests
        num_requests = 7
        for _ in range(num_requests):
            online_trainer.process_request(sample_context)

        stats = online_trainer.get_online_stats()

        # Check statistics are accurate
        assert stats["total_requests"] == num_requests
        assert stats["buffer_size"] == min(num_requests, 100)  # Buffer limit

        # Arm counts should be consistent
        arm_counts = stats["arm_counts"]
        assert torch.sum(arm_counts) <= num_requests  # Some may not have been updated yet