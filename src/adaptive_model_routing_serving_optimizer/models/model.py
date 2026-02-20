"""Core models for adaptive routing and contextual bandit algorithms."""

import logging
import math
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW


logger = logging.getLogger(__name__)


class ContextualBandit(nn.Module):
    """Contextual bandit for adaptive model routing decisions."""

    def __init__(self, context_dim: int, num_arms: int, config: Dict[str, Any]) -> None:
        """Initialize contextual bandit.

        Args:
            context_dim: Dimension of context vectors.
            num_arms: Number of arms (model variants).
            config: Configuration dictionary.
        """
        super().__init__()
        self.context_dim = context_dim
        self.num_arms = num_arms
        self.config = config
        self.algorithm = config["training"]["bandit"]["algorithm"]
        self.exploration_param = config["training"]["bandit"]["exploration_param"]

        # Neural network for reward prediction
        hidden_dim = max(64, context_dim)
        self.reward_network = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_arms)
        )

        # Uncertainty estimation network (for Thompson Sampling)
        self.uncertainty_network = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_arms)
        )

        # Statistics for UCB algorithm (register as buffers to move with model)
        self.register_buffer('arm_counts', torch.zeros(num_arms))
        self.register_buffer('arm_rewards', torch.zeros(num_arms))
        self.total_count = 0

        # Experience replay buffer
        self.memory_size = config["training"]["bandit"]["memory_size"]
        self.experience_buffer = deque(maxlen=self.memory_size)

        # Optimizer
        self.optimizer = AdamW(
            self.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"]
        )

        logger.info(f"Initialized ContextualBandit with {self.algorithm} algorithm")

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Forward pass through reward network.

        Args:
            context: Context tensor of shape (batch_size, context_dim).

        Returns:
            Predicted rewards of shape (batch_size, num_arms).
        """
        return self.reward_network(context)

    def select_arm(self, context: torch.Tensor) -> int:
        """Select arm based on current policy.

        Args:
            context: Context vector of shape (context_dim,).

        Returns:
            Selected arm index.
        """
        self.eval()
        with torch.no_grad():
            if len(context.shape) == 1:
                context = context.unsqueeze(0)

            if self.algorithm == "epsilon_greedy":
                return self._epsilon_greedy_selection(context)
            elif self.algorithm == "ucb":
                return self._ucb_selection(context)
            elif self.algorithm == "thompson_sampling":
                return self._thompson_sampling_selection(context)
            else:
                # Default to UCB
                return self._ucb_selection(context)

    def _epsilon_greedy_selection(self, context: torch.Tensor) -> int:
        """Epsilon-greedy arm selection.

        Args:
            context: Context tensor.

        Returns:
            Selected arm index.
        """
        if np.random.random() < self.exploration_param:
            return np.random.randint(self.num_arms)
        else:
            rewards = self.forward(context)
            return int(torch.argmax(rewards[0]).cpu().item())

    def _ucb_selection(self, context: torch.Tensor) -> int:
        """Upper Confidence Bound arm selection.

        Args:
            context: Context tensor.

        Returns:
            Selected arm index.
        """
        rewards = self.forward(context)[0]  # Remove batch dimension

        if self.total_count == 0:
            return np.random.randint(self.num_arms)

        # Calculate UCB values
        confidence_bounds = torch.sqrt(
            2 * math.log(self.total_count) / (self.arm_counts + 1e-8)
        ) * self.exploration_param

        ucb_values = rewards + confidence_bounds
        return int(torch.argmax(ucb_values).cpu().item())

    def _thompson_sampling_selection(self, context: torch.Tensor) -> int:
        """Thompson Sampling arm selection.

        Args:
            context: Context tensor.

        Returns:
            Selected arm index.
        """
        mean_rewards = self.forward(context)[0]
        uncertainties = torch.abs(self.uncertainty_network(context)[0])

        # Sample from posterior distributions
        sampled_rewards = torch.normal(mean_rewards, uncertainties + 1e-3)
        return int(torch.argmax(sampled_rewards).cpu().item())

    def update(self, context: torch.Tensor, arm: int, reward: float) -> None:
        """Update bandit with observed reward.

        Args:
            context: Context vector.
            arm: Selected arm index.
            reward: Observed reward.
        """
        # Store experience
        experience = (context.detach().cpu(), arm, reward)
        self.experience_buffer.append(experience)

        # Update statistics
        self.arm_counts[arm] += 1
        self.arm_rewards[arm] += reward
        self.total_count += 1

        # Train network periodically
        if len(self.experience_buffer) >= 32 and self.total_count % 10 == 0:
            self._train_networks()

    def _train_networks(self) -> None:
        """Train reward and uncertainty networks on experience buffer."""
        self.train()

        # Sample batch from experience buffer
        import random
        batch_size = min(32, len(self.experience_buffer))
        experiences = random.sample(list(self.experience_buffer), batch_size)

        # Get device from model parameters
        device = next(self.parameters()).device

        contexts = torch.stack([exp[0] for exp in experiences]).to(device)
        arms = torch.tensor([exp[1] for exp in experiences], dtype=torch.long, device=device)
        rewards = torch.tensor([exp[2] for exp in experiences], dtype=torch.float32, device=device)

        # Train reward network
        predicted_rewards = self.forward(contexts)
        selected_rewards = predicted_rewards[torch.arange(batch_size), arms]

        reward_loss = F.mse_loss(selected_rewards, rewards)

        # Train uncertainty network (simplified)
        uncertainties = self.uncertainty_network(contexts)
        selected_uncertainties = uncertainties[torch.arange(batch_size), arms]

        # Uncertainty should be higher for arms with higher prediction error
        prediction_errors = torch.abs(selected_rewards.detach() - rewards)
        uncertainty_loss = F.mse_loss(selected_uncertainties, prediction_errors)

        total_loss = reward_loss + 0.1 * uncertainty_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

    def get_arm_statistics(self) -> Dict[str, torch.Tensor]:
        """Get current arm statistics.

        Returns:
            Dictionary containing arm statistics.
        """
        average_rewards = self.arm_rewards / (self.arm_counts + 1e-8)
        return {
            "arm_counts": self.arm_counts.clone(),
            "arm_rewards": self.arm_rewards.clone(),
            "average_rewards": average_rewards,
            "total_count": self.total_count
        }


class RoutingPolicy(nn.Module):
    """Neural routing policy for model variant selection."""

    def __init__(self, context_dim: int, num_variants: int, config: Dict[str, Any]) -> None:
        """Initialize routing policy.

        Args:
            context_dim: Dimension of context features.
            num_variants: Number of model variants.
            config: Configuration dictionary.
        """
        super().__init__()
        self.context_dim = context_dim
        self.num_variants = num_variants
        self.config = config

        # Policy network
        hidden_dim = 128
        self.policy_network = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_variants)
        )

        # Value network for advantage estimation
        self.value_network = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

        logger.info(f"Initialized RoutingPolicy with {num_variants} variants")

    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through policy and value networks.

        Args:
            context: Context tensor of shape (batch_size, context_dim).

        Returns:
            Tuple of (policy_logits, state_values).
        """
        policy_logits = self.policy_network(context)
        state_values = self.value_network(context)
        return policy_logits, state_values

    def select_action(self, context: torch.Tensor, deterministic: bool = False) -> int:
        """Select action using current policy.

        Args:
            context: Context vector.
            deterministic: Whether to use deterministic policy.

        Returns:
            Selected action (variant index).
        """
        self.eval()
        with torch.no_grad():
            if len(context.shape) == 1:
                context = context.unsqueeze(0)

            logits, _ = self.forward(context)
            probs = F.softmax(logits, dim=-1)

            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = torch.multinomial(probs, num_samples=1).squeeze(-1)

            return int(action[0].cpu().item())

    def get_action_probabilities(self, context: torch.Tensor) -> torch.Tensor:
        """Get action probabilities for given context.

        Args:
            context: Context tensor.

        Returns:
            Action probabilities.
        """
        self.eval()
        with torch.no_grad():
            if len(context.shape) == 1:
                context = context.unsqueeze(0)

            logits, _ = self.forward(context)
            probs = F.softmax(logits, dim=-1)
            return probs[0]


class ModelVariantManager:
    """Manager for different model compression variants."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize model variant manager.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.variants = config["model"]["compression_variants"]
        self.architectures = config["model"]["architectures"]

        # Variant metadata
        self.variant_info = {}
        for i, variant in enumerate(self.variants):
            self.variant_info[i] = {
                "name": variant["type"],
                "memory_multiplier": variant["memory_multiplier"],
                "latency_multiplier": variant["latency_multiplier"],
                "precision": variant.get("precision", "float32"),
                "sparsity": variant.get("sparsity", 0.0),
            }

        # Performance cache
        self.performance_cache = defaultdict(dict)

        logger.info(f"Initialized ModelVariantManager with {len(self.variants)} variants")

    def get_variant_info(self, variant_idx: int) -> Dict[str, Any]:
        """Get information about a specific variant.

        Args:
            variant_idx: Variant index.

        Returns:
            Variant information dictionary.
        """
        return self.variant_info.get(variant_idx, {})

    def estimate_performance(
        self,
        variant_idx: int,
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate performance metrics for a variant given context.

        Args:
            variant_idx: Variant index.
            context: Context information.

        Returns:
            Estimated performance metrics.
        """
        variant = self.variant_info[variant_idx]
        batch_size = context.get("batch_size", 1)
        image_size = context.get("image_size", 224)

        # Base estimates (would be replaced with actual measurements)
        base_latency = 50.0  # ms
        base_memory = 1000.0  # MB
        base_accuracy = 0.95

        # Apply variant multipliers
        estimated_latency = (
            base_latency
            * variant["latency_multiplier"]
            * (batch_size / 32)
            * (image_size / 224) ** 2
        )

        estimated_memory = (
            base_memory
            * variant["memory_multiplier"]
            * batch_size
        )

        # Accuracy degradation for compressed variants
        accuracy_degradation = {
            "fp32": 0.0,
            "fp16": 0.002,
            "int8": 0.010,
            "pruned": 0.015
        }
        estimated_accuracy = base_accuracy - accuracy_degradation.get(
            variant["name"], 0.0
        )

        return {
            "latency_ms": estimated_latency,
            "memory_mb": estimated_memory,
            "accuracy": estimated_accuracy,
            "throughput_rps": 1000.0 / estimated_latency,
        }

    def calculate_reward(
        self,
        variant_idx: int,
        actual_performance: Dict[str, float],
        sla_constraints: Dict[str, float]
    ) -> float:
        """Calculate reward for routing decision.

        Args:
            variant_idx: Selected variant index.
            actual_performance: Actual measured performance.
            sla_constraints: SLA constraint thresholds.

        Returns:
            Reward value (higher is better).
        """
        reward = 0.0
        weights = self.config["routing"]["reward_weights"]

        # Latency reward (negative penalty for exceeding SLA)
        latency_ms = actual_performance.get("latency_ms", 0)
        latency_threshold = sla_constraints.get("p99_latency_ms", 50)

        if latency_ms <= latency_threshold:
            latency_reward = 1.0 - (latency_ms / latency_threshold)
        else:
            latency_reward = -((latency_ms - latency_threshold) / latency_threshold)

        reward += weights["latency"] * latency_reward

        # Accuracy reward
        accuracy = actual_performance.get("accuracy", 0)
        accuracy_threshold = sla_constraints.get("accuracy_threshold", 0.95)

        if accuracy >= accuracy_threshold:
            accuracy_reward = accuracy / accuracy_threshold
        else:
            accuracy_reward = -1.0  # Heavy penalty for accuracy violations

        reward += weights["accuracy"] * accuracy_reward

        # Memory efficiency reward
        memory_mb = actual_performance.get("memory_mb", 0)
        memory_threshold = sla_constraints.get("memory_limit_gb", 8) * 1024

        memory_reward = 1.0 - min(memory_mb / memory_threshold, 1.0)
        reward += weights["memory"] * memory_reward

        # Cost efficiency reward (inversely proportional to resource usage)
        variant = self.variant_info[variant_idx]
        cost_efficiency = 2.0 - variant["memory_multiplier"]  # Higher efficiency for compressed models
        reward += weights["cost"] * cost_efficiency

        return reward

    def get_best_variant_for_context(
        self,
        context: Dict[str, Any],
        sla_constraints: Dict[str, float]
    ) -> int:
        """Get best variant for given context (greedy baseline).

        Args:
            context: Context information.
            sla_constraints: SLA constraints.

        Returns:
            Best variant index.
        """
        best_variant = 0
        best_score = float("-inf")

        for i in range(len(self.variants)):
            performance = self.estimate_performance(i, context)
            score = self.calculate_reward(i, performance, sla_constraints)

            if score > best_score:
                best_score = score
                best_variant = i

        return best_variant


class AdaptiveRoutingModel(nn.Module):
    """Main adaptive routing model combining all components."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize adaptive routing model.

        Args:
            config: Configuration dictionary.
        """
        super().__init__()
        self.config = config
        self.context_dim = config["training"]["bandit"]["context_dim"]
        self.num_variants = len(config["model"]["compression_variants"])

        # Initialize components
        self.contextual_bandit = ContextualBandit(
            context_dim=self.context_dim,
            num_arms=self.num_variants,
            config=config
        )

        self.routing_policy = RoutingPolicy(
            context_dim=self.context_dim,
            num_variants=self.num_variants,
            config=config
        )

        self.variant_manager = ModelVariantManager(config)

        # Training state
        self.training_step = 0
        self.performance_history = []

        logger.info("Initialized AdaptiveRoutingModel")

    def select_model_variant(
        self,
        context: torch.Tensor,
        use_bandit: bool = True
    ) -> int:
        """Select model variant for given context.

        Args:
            context: Context vector.
            use_bandit: Whether to use contextual bandit or policy network.

        Returns:
            Selected variant index.
        """
        if use_bandit:
            return self.contextual_bandit.select_arm(context)
        else:
            return self.routing_policy.select_action(context)

    def update_with_feedback(
        self,
        context: torch.Tensor,
        variant: int,
        performance: Dict[str, float]
    ) -> None:
        """Update model with performance feedback.

        Args:
            context: Context vector used for decision.
            variant: Selected variant index.
            performance: Actual performance metrics.
        """
        # Calculate reward
        sla_constraints = self.config["routing"]["sla_constraints"]
        reward = self.variant_manager.calculate_reward(
            variant, performance, sla_constraints
        )

        # Update contextual bandit
        self.contextual_bandit.update(context, variant, reward)

        # Store for policy training
        self.performance_history.append({
            "context": context.clone(),
            "variant": variant,
            "reward": reward,
            "performance": performance.copy()
        })

        # Limit history size
        if len(self.performance_history) > 10000:
            self.performance_history.pop(0)

        self.training_step += 1

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get current routing statistics.

        Returns:
            Dictionary containing routing statistics.
        """
        stats = self.contextual_bandit.get_arm_statistics()
        stats.update({
            "training_steps": self.training_step,
            "history_size": len(self.performance_history),
        })

        if self.performance_history:
            recent_rewards = [h["reward"] for h in self.performance_history[-100:]]
            stats["average_recent_reward"] = np.mean(recent_rewards)

        return stats