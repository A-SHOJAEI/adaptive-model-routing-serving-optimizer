"""Tests for model components."""

import pytest
import torch
import numpy as np
from typing import Dict, Any

from src.adaptive_model_routing_serving_optimizer.models.model import (
    AdaptiveRoutingModel,
    ContextualBandit,
    RoutingPolicy,
    ModelVariantManager
)


class TestContextualBandit:
    """Test cases for ContextualBandit."""

    def test_initialization(self, test_config: Dict[str, Any]) -> None:
        """Test ContextualBandit initialization."""
        context_dim = test_config["training"]["bandit"]["context_dim"]
        num_arms = len(test_config["model"]["compression_variants"])

        bandit = ContextualBandit(context_dim, num_arms, test_config)

        assert bandit.context_dim == context_dim
        assert bandit.num_arms == num_arms
        assert bandit.algorithm == "ucb"

    def test_forward_pass(self, test_config: Dict[str, Any]) -> None:
        """Test forward pass through reward network."""
        context_dim = test_config["training"]["bandit"]["context_dim"]
        num_arms = len(test_config["model"]["compression_variants"])

        bandit = ContextualBandit(context_dim, num_arms, test_config)
        context = torch.randn(4, context_dim)  # Batch of 4

        rewards = bandit.forward(context)

        assert rewards.shape == (4, num_arms)
        assert torch.all(torch.isfinite(rewards))

    def test_select_arm(self, test_config: Dict[str, Any]) -> None:
        """Test arm selection."""
        context_dim = test_config["training"]["bandit"]["context_dim"]
        num_arms = len(test_config["model"]["compression_variants"])

        bandit = ContextualBandit(context_dim, num_arms, test_config)
        context = torch.randn(context_dim)

        arm = bandit.select_arm(context)

        assert isinstance(arm, int)
        assert 0 <= arm < num_arms

    def test_update_bandit(self, test_config: Dict[str, Any]) -> None:
        """Test bandit update with feedback."""
        context_dim = test_config["training"]["bandit"]["context_dim"]
        num_arms = len(test_config["model"]["compression_variants"])

        bandit = ContextualBandit(context_dim, num_arms, test_config)
        context = torch.randn(context_dim)

        # Initial counts should be zero
        initial_counts = bandit.arm_counts.clone()
        assert torch.all(initial_counts == 0)

        # Update with feedback
        arm = 0
        reward = 0.8
        bandit.update(context, arm, reward)

        # Check that counts and rewards are updated
        assert bandit.arm_counts[arm] == 1
        assert bandit.arm_rewards[arm] == reward
        assert bandit.total_count == 1

    def test_get_arm_statistics(self, test_config: Dict[str, Any]) -> None:
        """Test arm statistics retrieval."""
        context_dim = test_config["training"]["bandit"]["context_dim"]
        num_arms = len(test_config["model"]["compression_variants"])

        bandit = ContextualBandit(context_dim, num_arms, test_config)

        # Update with some feedback
        context = torch.randn(context_dim)
        bandit.update(context, 0, 0.8)
        bandit.update(context, 1, 0.6)

        stats = bandit.get_arm_statistics()

        assert "arm_counts" in stats
        assert "arm_rewards" in stats
        assert "average_rewards" in stats
        assert "total_count" in stats

        assert stats["total_count"] == 2

    def test_exploration_algorithms(self, test_config: Dict[str, Any]) -> None:
        """Test different exploration algorithms."""
        context_dim = test_config["training"]["bandit"]["context_dim"]
        num_arms = len(test_config["model"]["compression_variants"])

        algorithms = ["epsilon_greedy", "ucb", "thompson_sampling"]

        for algorithm in algorithms:
            config = test_config.copy()
            config["training"]["bandit"]["algorithm"] = algorithm

            bandit = ContextualBandit(context_dim, num_arms, config)
            context = torch.randn(context_dim)

            arm = bandit.select_arm(context)
            assert 0 <= arm < num_arms


class TestRoutingPolicy:
    """Test cases for RoutingPolicy."""

    def test_initialization(self, test_config: Dict[str, Any]) -> None:
        """Test RoutingPolicy initialization."""
        context_dim = test_config["training"]["bandit"]["context_dim"]
        num_variants = len(test_config["model"]["compression_variants"])

        policy = RoutingPolicy(context_dim, num_variants, test_config)

        assert policy.context_dim == context_dim
        assert policy.num_variants == num_variants

    def test_forward_pass(self, test_config: Dict[str, Any]) -> None:
        """Test forward pass through policy networks."""
        context_dim = test_config["training"]["bandit"]["context_dim"]
        num_variants = len(test_config["model"]["compression_variants"])

        policy = RoutingPolicy(context_dim, num_variants, test_config)
        context = torch.randn(4, context_dim)  # Batch of 4

        logits, values = policy.forward(context)

        assert logits.shape == (4, num_variants)
        assert values.shape == (4, 1)
        assert torch.all(torch.isfinite(logits))
        assert torch.all(torch.isfinite(values))

    def test_select_action(self, test_config: Dict[str, Any]) -> None:
        """Test action selection."""
        context_dim = test_config["training"]["bandit"]["context_dim"]
        num_variants = len(test_config["model"]["compression_variants"])

        policy = RoutingPolicy(context_dim, num_variants, test_config)
        context = torch.randn(context_dim)

        # Test stochastic selection
        action_stochastic = policy.select_action(context, deterministic=False)
        assert isinstance(action_stochastic, int)
        assert 0 <= action_stochastic < num_variants

        # Test deterministic selection
        action_deterministic = policy.select_action(context, deterministic=True)
        assert isinstance(action_deterministic, int)
        assert 0 <= action_deterministic < num_variants

        # Deterministic should be consistent
        action_deterministic2 = policy.select_action(context, deterministic=True)
        assert action_deterministic == action_deterministic2

    def test_get_action_probabilities(self, test_config: Dict[str, Any]) -> None:
        """Test action probability computation."""
        context_dim = test_config["training"]["bandit"]["context_dim"]
        num_variants = len(test_config["model"]["compression_variants"])

        policy = RoutingPolicy(context_dim, num_variants, test_config)
        context = torch.randn(context_dim)

        probs = policy.get_action_probabilities(context)

        assert probs.shape == (num_variants,)
        assert torch.all(probs >= 0)
        assert torch.all(probs <= 1)
        assert torch.allclose(torch.sum(probs), torch.tensor(1.0), atol=1e-6)


class TestModelVariantManager:
    """Test cases for ModelVariantManager."""

    def test_initialization(self, test_config: Dict[str, Any]) -> None:
        """Test ModelVariantManager initialization."""
        manager = ModelVariantManager(test_config)

        assert len(manager.variants) == len(test_config["model"]["compression_variants"])
        assert len(manager.variant_info) > 0

    def test_get_variant_info(self, test_config: Dict[str, Any]) -> None:
        """Test variant information retrieval."""
        manager = ModelVariantManager(test_config)

        info_0 = manager.get_variant_info(0)
        assert isinstance(info_0, dict)
        assert "name" in info_0
        assert "memory_multiplier" in info_0
        assert "latency_multiplier" in info_0

        # Test invalid variant
        info_invalid = manager.get_variant_info(999)
        assert info_invalid == {}

    def test_estimate_performance(self, test_config: Dict[str, Any]) -> None:
        """Test performance estimation."""
        manager = ModelVariantManager(test_config)

        context = {
            "batch_size": 4,
            "image_size": 224
        }

        performance = manager.estimate_performance(0, context)

        assert isinstance(performance, dict)
        assert "latency_ms" in performance
        assert "memory_mb" in performance
        assert "accuracy" in performance
        assert "throughput_rps" in performance

        # Check reasonable values
        assert performance["latency_ms"] > 0
        assert performance["memory_mb"] > 0
        assert 0 < performance["accuracy"] <= 1
        assert performance["throughput_rps"] > 0

    def test_calculate_reward(self, test_config: Dict[str, Any]) -> None:
        """Test reward calculation."""
        manager = ModelVariantManager(test_config)

        performance = {
            "latency_ms": 45.0,
            "memory_mb": 800.0,
            "accuracy": 0.96,
            "throughput_rps": 100.0
        }

        sla_constraints = test_config["routing"]["sla_constraints"]
        reward = manager.calculate_reward(0, performance, sla_constraints)

        assert isinstance(reward, float)
        assert torch.isfinite(torch.tensor(reward))

    def test_get_best_variant_for_context(self, test_config: Dict[str, Any]) -> None:
        """Test best variant selection."""
        manager = ModelVariantManager(test_config)

        context = {
            "batch_size": 1,
            "image_size": 224,
            "priority": 3
        }

        sla_constraints = test_config["routing"]["sla_constraints"]
        best_variant = manager.get_best_variant_for_context(context, sla_constraints)

        assert isinstance(best_variant, int)
        assert 0 <= best_variant < len(manager.variants)

    def test_reward_calculation_consistency(self, test_config: Dict[str, Any]) -> None:
        """Test that reward calculation is consistent."""
        manager = ModelVariantManager(test_config)

        performance = {
            "latency_ms": 40.0,
            "memory_mb": 1000.0,
            "accuracy": 0.97
        }

        sla_constraints = test_config["routing"]["sla_constraints"]

        # Calculate reward multiple times
        reward1 = manager.calculate_reward(0, performance, sla_constraints)
        reward2 = manager.calculate_reward(0, performance, sla_constraints)

        assert reward1 == reward2


class TestAdaptiveRoutingModel:
    """Test cases for AdaptiveRoutingModel."""

    def test_initialization(self, adaptive_model: AdaptiveRoutingModel) -> None:
        """Test AdaptiveRoutingModel initialization."""
        assert hasattr(adaptive_model, 'contextual_bandit')
        assert hasattr(adaptive_model, 'routing_policy')
        assert hasattr(adaptive_model, 'variant_manager')

        assert adaptive_model.context_dim == 16
        assert adaptive_model.num_variants == 2

    def test_select_model_variant(
        self,
        adaptive_model: AdaptiveRoutingModel,
        sample_context: torch.Tensor
    ) -> None:
        """Test model variant selection."""
        # Test bandit selection
        variant_bandit = adaptive_model.select_model_variant(sample_context, use_bandit=True)
        assert isinstance(variant_bandit, int)
        assert 0 <= variant_bandit < adaptive_model.num_variants

        # Test policy selection
        variant_policy = adaptive_model.select_model_variant(sample_context, use_bandit=False)
        assert isinstance(variant_policy, int)
        assert 0 <= variant_policy < adaptive_model.num_variants

    def test_update_with_feedback(
        self,
        adaptive_model: AdaptiveRoutingModel,
        sample_context: torch.Tensor
    ) -> None:
        """Test feedback update."""
        variant = 0
        performance = {
            "latency_ms": 42.0,
            "memory_mb": 900.0,
            "accuracy": 0.96
        }

        initial_step = adaptive_model.training_step
        adaptive_model.update_with_feedback(sample_context, variant, performance)

        assert adaptive_model.training_step == initial_step + 1
        assert len(adaptive_model.performance_history) == 1

    def test_get_routing_statistics(self, adaptive_model: AdaptiveRoutingModel) -> None:
        """Test routing statistics retrieval."""
        stats = adaptive_model.get_routing_statistics()

        assert isinstance(stats, dict)
        assert "training_steps" in stats
        assert "history_size" in stats
        assert "arm_counts" in stats

    def test_model_training_mode(self, adaptive_model: AdaptiveRoutingModel) -> None:
        """Test model training/eval mode switching."""
        # Test training mode
        adaptive_model.train()
        assert adaptive_model.training
        assert adaptive_model.contextual_bandit.training
        assert adaptive_model.routing_policy.training

        # Test evaluation mode
        adaptive_model.eval()
        assert not adaptive_model.training
        assert not adaptive_model.contextual_bandit.training
        assert not adaptive_model.routing_policy.training

    def test_model_device_placement(self, adaptive_model: AdaptiveRoutingModel) -> None:
        """Test model device placement."""
        device = torch.device("cpu")

        # Move to device
        adaptive_model.to(device)

        # Check that components are on correct device
        for param in adaptive_model.contextual_bandit.parameters():
            assert param.device == device

        for param in adaptive_model.routing_policy.parameters():
            assert param.device == device

    def test_state_dict_consistency(self, adaptive_model: AdaptiveRoutingModel) -> None:
        """Test that state dict can be saved and loaded."""
        # Save state
        state_dict = adaptive_model.state_dict()

        # Create new model
        new_model = AdaptiveRoutingModel(adaptive_model.config)

        # Load state
        new_model.load_state_dict(state_dict)

        # Check that parameters are the same
        for (name1, param1), (name2, param2) in zip(
            adaptive_model.named_parameters(),
            new_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2, atol=1e-6)

    def test_performance_history_limits(
        self,
        adaptive_model: AdaptiveRoutingModel,
        sample_context: torch.Tensor
    ) -> None:
        """Test that performance history is limited in size."""
        performance = {
            "latency_ms": 45.0,
            "memory_mb": 800.0,
            "accuracy": 0.95
        }

        # Add many updates
        for i in range(15000):  # More than the 10000 limit
            adaptive_model.update_with_feedback(sample_context, 0, performance)

        # Should be limited to 10000
        assert len(adaptive_model.performance_history) == 10000

    def test_gradient_computation(
        self,
        adaptive_model: AdaptiveRoutingModel,
        sample_batch_context: torch.Tensor
    ) -> None:
        """Test that gradients can be computed."""
        adaptive_model.train()

        # Forward pass through bandit
        rewards = adaptive_model.contextual_bandit(sample_batch_context)
        loss_bandit = rewards.mean()

        # Forward pass through policy
        logits, values = adaptive_model.routing_policy(sample_batch_context)
        loss_policy = logits.mean() + values.mean()

        # Check gradients can be computed
        loss_bandit.backward(retain_graph=True)
        loss_policy.backward()

        # Check that gradients exist and are finite
        for param in adaptive_model.contextual_bandit.parameters():
            if param.grad is not None:
                assert torch.all(torch.isfinite(param.grad))

        for param in adaptive_model.routing_policy.parameters():
            if param.grad is not None:
                assert torch.all(torch.isfinite(param.grad))