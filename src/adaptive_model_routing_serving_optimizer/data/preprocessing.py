"""Request preprocessing and context extraction utilities."""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import psutil
import GPUtil


logger = logging.getLogger(__name__)


class RequestPreprocessor:
    """Preprocessor for incoming inference requests."""

    def __init__(self, config: Dict[str, Any], device: Optional[torch.device] = None) -> None:
        """Initialize request preprocessor.

        Args:
            config: Configuration dictionary.
            device: Device to create tensors on (cuda/cpu).
        """
        self.config = config
        self.context_dim = config["training"]["bandit"]["context_dim"]
        self.features = config["routing"]["features"]
        self.device = device or torch.device("cpu")

        # Historical data for moving averages
        self.latency_history: List[float] = []
        self.memory_history: List[float] = []
        self.queue_history: List[int] = []

        # Feature normalizers (will be fitted during training)
        self.feature_stats = {
            "mean": torch.zeros(self.context_dim, device=self.device),
            "std": torch.ones(self.context_dim, device=self.device),
        }

        logger.info(f"Initialized RequestPreprocessor with {len(self.features)} features on device: {self.device}")

    def extract_features(self, request_data: Dict[str, Any]) -> torch.Tensor:
        """Extract features from incoming request.

        Args:
            request_data: Dictionary containing request information.

        Returns:
            Feature tensor of shape (context_dim,).
        """
        features = []

        # Extract basic request features
        request_complexity = self._calculate_request_complexity(request_data)
        features.append(request_complexity)

        # Historical latency (moving average)
        historical_latency = np.mean(self.latency_history[-100:]) if self.latency_history else 50.0
        features.append(historical_latency)

        # Current system state
        gpu_memory_usage = self._get_gpu_memory_usage()
        features.append(gpu_memory_usage)

        queue_length = request_data.get("queue_length", 0)
        features.append(queue_length)

        # Model load metrics
        model_load = self._calculate_model_load()
        features.append(model_load)

        # Temporal features
        current_time = time.time()
        hour_of_day = (current_time % 86400) / 3600  # Normalize to 0-24
        features.append(hour_of_day)

        day_of_week = ((current_time // 86400) % 7) / 7  # Normalize to 0-1
        features.append(day_of_week)

        # User/request specific features
        user_priority = request_data.get("priority", 3) / 5.0  # Normalize to 0-1
        features.append(user_priority)

        accuracy_requirement = request_data.get("accuracy_requirement", 0.95)
        features.append(accuracy_requirement)

        # Pad or truncate to context_dim
        while len(features) < self.context_dim:
            features.append(0.0)
        features = features[:self.context_dim]

        feature_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)

        # Apply normalization
        feature_tensor = (feature_tensor - self.feature_stats["mean"]) / (self.feature_stats["std"] + 1e-8)

        return feature_tensor

    def _calculate_request_complexity(self, request_data: Dict[str, Any]) -> float:
        """Calculate complexity score for the request.

        Args:
            request_data: Request information.

        Returns:
            Complexity score (0-100).
        """
        # Simple heuristic based on image size, batch size, etc.
        batch_size = request_data.get("batch_size", 1)
        image_size = request_data.get("image_size", 224)

        # Complexity increases with batch size and image resolution
        complexity = (batch_size * (image_size / 224) ** 2) * 10
        return min(complexity, 100.0)

    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage.

        Returns:
            Memory usage as fraction (0-1).
        """
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryUtil
            return 0.5  # Default if no GPU info available
        except Exception:
            return 0.5

    def _calculate_model_load(self) -> float:
        """Calculate current model load.

        Returns:
            Model load as fraction (0-1).
        """
        try:
            # Use CPU usage as a proxy for model load
            cpu_usage = psutil.cpu_percent(interval=None)
            return cpu_usage / 100.0
        except Exception:
            return 0.5

    def update_history(self, latency: float, memory_usage: float, queue_length: int) -> None:
        """Update historical metrics.

        Args:
            latency: Latest latency measurement.
            memory_usage: Latest memory usage.
            queue_length: Latest queue length.
        """
        max_history = 1000

        self.latency_history.append(latency)
        if len(self.latency_history) > max_history:
            self.latency_history.pop(0)

        self.memory_history.append(memory_usage)
        if len(self.memory_history) > max_history:
            self.memory_history.pop(0)

        self.queue_history.append(queue_length)
        if len(self.queue_history) > max_history:
            self.queue_history.pop(0)

    def fit_normalizers(self, features: torch.Tensor) -> None:
        """Fit feature normalizers on training data.

        Args:
            features: Training features tensor of shape (num_samples, context_dim).
        """
        self.feature_stats["mean"] = features.mean(dim=0)
        self.feature_stats["std"] = features.std(dim=0)

        logger.info("Fitted feature normalizers on training data")


class ContextExtractor:
    """Extractor for contextual bandit features from system and request state."""

    def __init__(self, config: Dict[str, Any], device: Optional[torch.device] = None) -> None:
        """Initialize context extractor.

        Args:
            config: Configuration dictionary.
            device: Device to create tensors on (cuda/cpu).
        """
        self.config = config
        self.context_dim = config["training"]["bandit"]["context_dim"]
        self.device = device or torch.device("cpu")

        # System monitors
        self.system_monitor = SystemMonitor(config)

        logger.info(f"Initialized ContextExtractor on device: {self.device}")

    def extract_context(
        self,
        request_data: Dict[str, Any],
        system_state: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Extract context vector for contextual bandit.

        Args:
            request_data: Information about the current request.
            system_state: Current system state (optional, will be queried if None).

        Returns:
            Context vector of shape (context_dim,).
        """
        if system_state is None:
            system_state = self.system_monitor.get_current_state()

        context_features = []

        # Request features
        context_features.extend([
            request_data.get("batch_size", 1) / 64.0,  # Normalize by max batch
            request_data.get("image_size", 224) / 512.0,  # Normalize by reasonable max
            request_data.get("priority", 3) / 5.0,  # Normalize priority
            request_data.get("accuracy_requirement", 0.95),
        ])

        # System features
        context_features.extend([
            system_state.get("gpu_memory_usage", 0.5),
            system_state.get("cpu_usage", 0.5),
            system_state.get("gpu_utilization", 0.5),
            system_state.get("queue_length", 0) / 100.0,  # Normalize by reasonable max
        ])

        # Temporal features
        current_time = time.time()
        context_features.extend([
            (current_time % 86400) / 86400,  # Time of day (0-1)
            ((current_time // 86400) % 7) / 7,  # Day of week (0-1)
            np.sin(2 * np.pi * current_time / 3600),  # Hourly periodicity
            np.cos(2 * np.pi * current_time / 3600),
        ])

        # SLA constraint features
        sla_constraints = self.config["routing"]["sla_constraints"]
        context_features.extend([
            sla_constraints["p99_latency_ms"] / 1000.0,  # Normalize to seconds
            sla_constraints["accuracy_threshold"],
            sla_constraints["memory_limit_gb"] / 32.0,  # Normalize by reasonable max
        ])

        # Pad to context_dim
        while len(context_features) < self.context_dim:
            context_features.append(0.0)

        context_tensor = torch.tensor(
            context_features[:self.context_dim],
            dtype=torch.float32,
            device=self.device
        )

        return context_tensor

    def extract_batch_context(
        self,
        batch_requests: List[Dict[str, Any]],
        system_state: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Extract context for a batch of requests.

        Args:
            batch_requests: List of request data dictionaries.
            system_state: Current system state.

        Returns:
            Context tensor of shape (batch_size, context_dim).
        """
        contexts = []
        for request_data in batch_requests:
            context = self.extract_context(request_data, system_state)
            contexts.append(context)

        return torch.stack(contexts)


class SystemMonitor:
    """Monitor for system metrics and performance."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize system monitor.

        Args:
            config: Configuration dictionary.
        """
        self.config = config

    def get_current_state(self) -> Dict[str, float]:
        """Get current system state.

        Returns:
            Dictionary containing current system metrics.
        """
        state = {}

        try:
            # GPU metrics
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                state["gpu_memory_usage"] = gpu.memoryUtil
                state["gpu_utilization"] = gpu.load
                state["gpu_temperature"] = gpu.temperature
            else:
                state["gpu_memory_usage"] = 0.5
                state["gpu_utilization"] = 0.5
                state["gpu_temperature"] = 65.0

            # CPU metrics
            state["cpu_usage"] = psutil.cpu_percent(interval=None) / 100.0

            # Memory metrics
            memory = psutil.virtual_memory()
            state["memory_usage"] = memory.percent / 100.0
            state["memory_available_gb"] = memory.available / (1024**3)

            # Network metrics (simplified)
            state["network_latency_ms"] = 10.0  # Placeholder

            # Queue metrics (would come from actual serving system)
            state["queue_length"] = 0  # Placeholder

        except Exception as e:
            logger.warning(f"Error getting system state: {e}")
            # Return default values
            state = {
                "gpu_memory_usage": 0.5,
                "gpu_utilization": 0.5,
                "gpu_temperature": 65.0,
                "cpu_usage": 0.5,
                "memory_usage": 0.5,
                "memory_available_gb": 8.0,
                "network_latency_ms": 10.0,
                "queue_length": 0,
            }

        return state

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance-related metrics.

        Returns:
            Dictionary containing performance metrics.
        """
        return {
            "latency_p99": 45.0,  # Placeholder - would come from actual metrics
            "throughput_rps": 100.0,
            "error_rate": 0.001,
            "cost_per_inference": 0.01,
        }