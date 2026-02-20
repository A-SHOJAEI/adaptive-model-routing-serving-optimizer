"""Adaptive Model Routing Serving Optimizer.

This package provides an intelligent model-serving gateway that dynamically routes
inference requests across multiple compression variants based on real-time latency SLOs,
GPU memory pressure, and request-level accuracy sensitivity.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"

# Lazy imports to avoid circular import issues and missing dependency errors
try:
    from .models.model import AdaptiveRoutingModel
    from .training.trainer import RoutingTrainer
    from .evaluation.metrics import RoutingMetrics

    __all__ = [
        "AdaptiveRoutingModel",
        "RoutingTrainer",
        "RoutingMetrics",
    ]
except ImportError:
    # Package is importable even without dependencies
    __all__ = []