"""Training pipeline for adaptive routing models."""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

# MLflow imports with error handling
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available. Training will proceed without tracking.")

from ..models.model import AdaptiveRoutingModel
from ..data.loader import create_data_loaders
from ..data.preprocessing import ContextExtractor, RequestPreprocessor


logger = logging.getLogger(__name__)


class RoutingTrainer:
    """Trainer for adaptive routing models with offline and online learning."""

    def __init__(
        self,
        model: AdaptiveRoutingModel,
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ) -> None:
        """Initialize trainer.

        Args:
            model: Adaptive routing model to train.
            config: Configuration dictionary.
            device: Device to use for training.
        """
        self.model = model
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Move model to device
        self.model.to(self.device)

        # Training configuration
        self.epochs = config["training"]["epochs"]
        self.learning_rate = config["training"]["learning_rate"]
        self.weight_decay = config["training"]["weight_decay"]
        self.patience = config["training"]["patience"]
        self.min_delta = config["training"]["min_delta"]
        self.gradient_clip_norm = config["training"]["gradient_clip_norm"]

        # Initialize optimizers
        self.bandit_optimizer = AdamW(
            self.model.contextual_bandit.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        self.policy_optimizer = AdamW(
            self.model.routing_policy.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Learning rate schedulers
        self.bandit_scheduler = ReduceLROnPlateau(
            self.bandit_optimizer, mode="min", patience=5, factor=0.5
        )

        self.policy_scheduler = CosineAnnealingLR(
            self.policy_optimizer, T_max=self.epochs
        )

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.training_history = []

        # Data components
        self.context_extractor = ContextExtractor(config, device=self.device)
        self.request_preprocessor = RequestPreprocessor(config, device=self.device)

        # MLflow setup
        self.use_mlflow = MLFLOW_AVAILABLE and config.get("mlflow", {}).get("tracking_uri")
        if self.use_mlflow:
            self._setup_mlflow()

        logger.info(f"Initialized RoutingTrainer on device: {self.device}")

    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        try:
            mlflow_config = self.config["mlflow"]
            mlflow.set_tracking_uri(mlflow_config["tracking_uri"])

            experiment_name = mlflow_config["experiment_name"]
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    mlflow.create_experiment(experiment_name)
            except Exception:
                mlflow.create_experiment(experiment_name)

            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow tracking configured: {experiment_name}")

        except Exception as e:
            logger.warning(f"Failed to setup MLflow: {e}")
            self.use_mlflow = False

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        save_dir: str = "checkpoints"
    ) -> Dict[str, List[float]]:
        """Train the adaptive routing model.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            save_dir: Directory to save checkpoints.

        Returns:
            Training history dictionary.
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        # Start MLflow run
        if self.use_mlflow:
            try:
                run_name = f"{self.config['mlflow']['run_name_prefix']}-{int(time.time())}"
                mlflow.start_run(run_name=run_name)
                mlflow.log_params({
                    "epochs": self.epochs,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.config["data"]["batch_size"],
                    "context_dim": self.config["training"]["bandit"]["context_dim"],
                    "num_variants": len(self.config["model"]["compression_variants"]),
                })
            except Exception as e:
                logger.warning(f"Failed to start MLflow run: {e}")

        logger.info("Starting training...")

        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # Train for one epoch
            train_metrics = self._train_epoch(train_loader)

            # Validate
            val_metrics = self._validate_epoch(val_loader)

            # Update learning rate schedulers
            self.bandit_scheduler.step(val_metrics["val_loss"])
            self.policy_scheduler.step()

            # Log metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            self.training_history.append(epoch_metrics)

            # Log to MLflow
            if self.use_mlflow:
                try:
                    mlflow.log_metrics(epoch_metrics, step=epoch)
                except Exception as e:
                    logger.warning(f"Failed to log MLflow metrics: {e}")

            # Print progress
            logger.info(
                f"Epoch {epoch + 1}/{self.epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Regret: {val_metrics.get('val_regret', 0):.4f}"
            )

            # Early stopping check
            if val_metrics["val_loss"] < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_metrics["val_loss"]
                self.patience_counter = 0

                # Save best model
                self._save_checkpoint(save_path / "best_model.pth", epoch, epoch_metrics)

            else:
                self.patience_counter += 1

            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(
                    save_path / f"checkpoint_epoch_{epoch + 1}.pth",
                    epoch,
                    epoch_metrics
                )

            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Save final model
        self._save_checkpoint(save_path / "final_model.pth", epoch, epoch_metrics)

        # End MLflow run
        if self.use_mlflow:
            try:
                # Log model
                mlflow.pytorch.log_model(
                    self.model,
                    "model",
                    registered_model_name="adaptive-routing-model"
                )
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"Failed to end MLflow run: {e}")

        logger.info("Training completed!")
        return {"history": self.training_history}

    def _train_epoch(self, train_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader.

        Returns:
            Training metrics for the epoch.
        """
        self.model.train()

        total_loss = 0.0
        total_bandit_loss = 0.0
        total_policy_loss = 0.0
        total_regret = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, (images, contexts) in enumerate(progress_bar):
            # Move to device
            images = images.to(self.device)
            batch_contexts = self._process_context_batch(contexts)

            # Simulate routing decisions and rewards
            batch_size = images.shape[0]
            variants = []
            rewards = []
            oracle_variants = []

            for i in range(batch_size):
                context = batch_contexts[i]

                # Get model decision
                variant = self.model.select_model_variant(context, use_bandit=True)
                variants.append(variant)

                # Get oracle decision (best possible)
                request_data = {"batch_size": 1, "image_size": 224, "priority": 3}
                sla_constraints = self.config["routing"]["sla_constraints"]
                oracle_variant = self.model.variant_manager.get_best_variant_for_context(
                    request_data, sla_constraints
                )
                oracle_variants.append(oracle_variant)

                # Simulate performance and calculate reward
                performance = self.model.variant_manager.estimate_performance(
                    variant, request_data
                )
                # Add some noise to make it more realistic
                performance["latency_ms"] += np.random.normal(0, 5)
                performance["accuracy"] += np.random.normal(0, 0.005)

                reward = self.model.variant_manager.calculate_reward(
                    variant, performance, sla_constraints
                )
                rewards.append(reward)

                # Update bandit with feedback
                self.model.contextual_bandit.update(context, variant, reward)

            variants = torch.tensor(variants, dtype=torch.long).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            oracle_variants = torch.tensor(oracle_variants, dtype=torch.long).to(self.device)

            # Train policy network (imitation learning on oracle actions)
            policy_logits, values = self.model.routing_policy(batch_contexts)

            # Policy loss (cross-entropy with oracle actions)
            policy_loss = F.cross_entropy(policy_logits, oracle_variants)

            # Value loss (predict average reward)
            value_targets = rewards.unsqueeze(1)
            value_loss = F.mse_loss(values, value_targets)

            policy_loss_batch = policy_loss + 0.1 * value_loss

            # Bandit loss (contextual bandit learning)
            predicted_rewards = self.model.contextual_bandit(batch_contexts)
            selected_rewards = predicted_rewards[torch.arange(batch_size), variants]
            bandit_loss = F.mse_loss(selected_rewards, rewards)

            # Combined loss
            total_loss_batch = policy_loss_batch + bandit_loss

            # Backward pass for policy
            self.policy_optimizer.zero_grad()
            policy_loss_batch.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(
                self.model.routing_policy.parameters(),
                self.gradient_clip_norm
            )
            self.policy_optimizer.step()

            # Backward pass for bandit
            self.bandit_optimizer.zero_grad()
            bandit_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.contextual_bandit.parameters(),
                self.gradient_clip_norm
            )
            self.bandit_optimizer.step()

            # Calculate regret (difference from oracle)
            oracle_rewards = []
            for i, oracle_var in enumerate(oracle_variants):
                oracle_performance = self.model.variant_manager.estimate_performance(
                    oracle_var.cpu().item(), {"batch_size": 1, "image_size": 224, "priority": 3}
                )
                oracle_reward = self.model.variant_manager.calculate_reward(
                    oracle_var.cpu().item(), oracle_performance, sla_constraints
                )
                oracle_rewards.append(oracle_reward)

            oracle_rewards = torch.tensor(oracle_rewards).to(self.device)
            regret = torch.mean(oracle_rewards - rewards)

            # Accumulate metrics
            total_loss += total_loss_batch.detach().cpu().item()
            total_bandit_loss += bandit_loss.detach().cpu().item()
            total_policy_loss += policy_loss_batch.detach().cpu().item()
            total_regret += regret.detach().cpu().item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                "Loss": f"{total_loss_batch.detach().cpu().item():.4f}",
                "Regret": f"{regret.detach().cpu().item():.4f}"
            })

        return {
            "train_loss": total_loss / num_batches,
            "train_bandit_loss": total_bandit_loss / num_batches,
            "train_policy_loss": total_policy_loss / num_batches,
            "train_regret": total_regret / num_batches,
        }

    def _validate_epoch(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Validate for one epoch.

        Args:
            val_loader: Validation data loader.

        Returns:
            Validation metrics for the epoch.
        """
        self.model.eval()

        total_loss = 0.0
        total_regret = 0.0
        total_accuracy = 0.0
        num_batches = 0

        with torch.no_grad():
            for images, contexts in val_loader:
                images = images.to(self.device)
                batch_contexts = self._process_context_batch(contexts)

                # Simulate validation routing
                batch_size = images.shape[0]
                variants = []
                rewards = []
                oracle_variants = []

                for i in range(batch_size):
                    context = batch_contexts[i]

                    # Get model decision (deterministic for validation)
                    variant = self.model.routing_policy.select_action(
                        context, deterministic=True
                    )
                    variants.append(variant)

                    # Get oracle decision
                    request_data = {"batch_size": 1, "image_size": 224, "priority": 3}
                    sla_constraints = self.config["routing"]["sla_constraints"]
                    oracle_variant = self.model.variant_manager.get_best_variant_for_context(
                        request_data, sla_constraints
                    )
                    oracle_variants.append(oracle_variant)

                    # Calculate reward
                    performance = self.model.variant_manager.estimate_performance(
                        variant, request_data
                    )
                    reward = self.model.variant_manager.calculate_reward(
                        variant, performance, sla_constraints
                    )
                    rewards.append(reward)

                variants = torch.tensor(variants, dtype=torch.long).to(self.device)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
                oracle_variants = torch.tensor(oracle_variants, dtype=torch.long).to(self.device)

                # Calculate losses
                policy_logits, values = self.model.routing_policy(batch_contexts)
                policy_loss = F.cross_entropy(policy_logits, oracle_variants)
                value_loss = F.mse_loss(values, rewards.unsqueeze(1))

                total_loss += (policy_loss + 0.1 * value_loss).detach().cpu().item()

                # Calculate regret
                oracle_rewards = []
                for oracle_var in oracle_variants:
                    oracle_performance = self.model.variant_manager.estimate_performance(
                        oracle_var.cpu().item(), {"batch_size": 1, "image_size": 224, "priority": 3}
                    )
                    oracle_reward = self.model.variant_manager.calculate_reward(
                        oracle_var.cpu().item(), oracle_performance, sla_constraints
                    )
                    oracle_rewards.append(oracle_reward)

                oracle_rewards = torch.tensor(oracle_rewards).to(self.device)
                regret = torch.mean(oracle_rewards - rewards)
                total_regret += regret.detach().cpu().item()

                # Calculate routing accuracy
                correct_variants = (variants == oracle_variants).float().mean()
                total_accuracy += correct_variants.detach().cpu().item()

                num_batches += 1

        return {
            "val_loss": total_loss / num_batches,
            "val_regret": total_regret / num_batches,
            "val_routing_accuracy": total_accuracy / num_batches,
        }

    def _process_context_batch(
        self,
        context_batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Process a batch of contexts into feature vectors.

        Args:
            context_batch: Dictionary of context tensors.

        Returns:
            Processed context tensor.
        """
        batch_size = next(iter(context_batch.values())).shape[0]
        contexts = []

        for i in range(batch_size):
            # Extract single context
            single_context = {k: v[i] for k, v in context_batch.items()}

            # Convert to request format
            request_data = {
                "batch_size": 1,
                "image_size": 224,
                "priority": int(single_context.get("user_priority", torch.tensor(3)).cpu().item()),
                "accuracy_requirement": float(single_context.get("accuracy_requirement", torch.tensor(0.95)).cpu().item()),
            }

            # Extract features
            context_vector = self.context_extractor.extract_context(request_data)
            contexts.append(context_vector)

        # Contexts are already on the correct device from context_extractor
        return torch.stack(contexts)

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
            epoch: Current epoch.
            metrics: Current metrics.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "bandit_optimizer_state_dict": self.bandit_optimizer.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "bandit_scheduler_state_dict": self.bandit_scheduler.state_dict(),
            "policy_scheduler_state_dict": self.policy_scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file.

        Returns:
            Loaded checkpoint data.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.bandit_optimizer.load_state_dict(checkpoint["bandit_optimizer_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self.bandit_scheduler.load_state_dict(checkpoint["bandit_scheduler_state_dict"])
        self.policy_scheduler.load_state_dict(checkpoint["policy_scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")

        return checkpoint


class OnlineTrainer:
    """Online trainer for real-time adaptation."""

    def __init__(
        self,
        model: AdaptiveRoutingModel,
        config: Dict[str, Any]
    ) -> None:
        """Initialize online trainer.

        Args:
            model: Adaptive routing model.
            config: Configuration dictionary.
        """
        self.model = model
        self.config = config
        self.update_frequency = config["training"]["bandit"]["update_frequency"]

        # Online statistics
        self.total_requests = 0
        self.performance_buffer = []

        logger.info("Initialized OnlineTrainer")

    def process_request(
        self,
        context: torch.Tensor,
        performance_feedback: Optional[Dict[str, float]] = None
    ) -> int:
        """Process an online request and potentially update the model.

        Args:
            context: Request context.
            performance_feedback: Performance metrics from previous request.

        Returns:
            Selected model variant.
        """
        # Update model with feedback from previous request if available
        if performance_feedback is not None and len(self.performance_buffer) > 0:
            prev_context, prev_variant = self.performance_buffer[-1]
            self.model.update_with_feedback(prev_context, prev_variant, performance_feedback)

        # Select variant for current request
        variant = self.model.select_model_variant(context, use_bandit=True)

        # Store for next update
        self.performance_buffer.append((context.clone(), variant))
        if len(self.performance_buffer) > 100:  # Keep limited history
            self.performance_buffer.pop(0)

        self.total_requests += 1

        return variant

    def get_online_stats(self) -> Dict[str, Any]:
        """Get online training statistics.

        Returns:
            Dictionary of online statistics.
        """
        stats = self.model.get_routing_statistics()
        stats.update({
            "total_requests": self.total_requests,
            "buffer_size": len(self.performance_buffer),
        })
        return stats