"""Data loading utilities for model zoo and synthetic data generation."""

import logging
import random
import time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import (
    resnet50, efficientnet_b0, mobilenet_v2,
    vit_b_16, ResNet50_Weights, EfficientNet_B0_Weights,
    MobileNet_V2_Weights, ViT_B_16_Weights
)

logger = logging.getLogger(__name__)


class ModelZooLoader:
    """Loader for TensorRT Model Zoo and PyTorch pretrained models."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the model zoo loader.

        Args:
            config: Configuration dictionary containing model specifications.
        """
        self.config = config
        self.architectures = config["model"]["architectures"]
        self.compression_variants = config["model"]["compression_variants"]
        self.device = torch.device(config["hardware"]["device"])

        # Model registry
        self.model_registry = {
            "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
            "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1),
            "mobilenet_v2": (mobilenet_v2, MobileNet_V2_Weights.IMAGENET1K_V2),
            "vit_base_patch16_224": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1),
        }

        logger.info(f"Initialized ModelZooLoader with {len(self.architectures)} architectures")

    def load_model(self, architecture: str, variant: str = "fp32") -> torch.nn.Module:
        """Load a specific model architecture with compression variant.

        Args:
            architecture: Model architecture name.
            variant: Compression variant (fp32, fp16, int8, pruned).

        Returns:
            Loaded PyTorch model.

        Raises:
            ValueError: If architecture is not supported.
        """
        if architecture not in self.model_registry:
            raise ValueError(f"Unsupported architecture: {architecture}")

        model_fn, weights = self.model_registry[architecture]
        model = model_fn(weights=weights)

        # Apply compression variant
        model = self._apply_compression(model, variant)
        model.eval()
        model.to(self.device)

        # Compile for optimization if enabled
        if self.config["hardware"].get("compile_model", False):
            model = torch.compile(model)

        logger.info(f"Loaded {architecture} with {variant} compression")
        return model

    def _apply_compression(self, model: torch.nn.Module, variant: str) -> torch.nn.Module:
        """Apply compression technique to model.

        Args:
            model: PyTorch model to compress.
            variant: Compression variant to apply.

        Returns:
            Compressed model.
        """
        if variant == "fp16":
            model = model.half()
        elif variant == "int8":
            # Simulated int8 quantization
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
            )
        elif variant == "pruned":
            # Simulated pruning by zeroing out random weights
            with torch.no_grad():
                for param in model.parameters():
                    if param.dim() > 1:  # Only prune weight matrices/kernels
                        mask = torch.rand_like(param) > 0.5  # 50% sparsity
                        param.data *= mask.float()

        return model

    def get_model_variants(self, architecture: str) -> Dict[str, torch.nn.Module]:
        """Get all compression variants of a model architecture.

        Args:
            architecture: Model architecture name.

        Returns:
            Dictionary mapping variant names to loaded models.
        """
        variants = {}
        for variant_config in self.compression_variants:
            variant_name = variant_config["type"]
            try:
                variants[variant_name] = self.load_model(architecture, variant_name)
            except Exception as e:
                logger.warning(f"Failed to load {architecture}-{variant_name}: {e}")

        return variants

    def get_all_model_variants(self) -> Dict[str, Dict[str, torch.nn.Module]]:
        """Get all model architectures with all variants.

        Returns:
            Nested dictionary: {architecture: {variant: model}}
        """
        all_models = {}
        for arch in self.architectures:
            all_models[arch] = self.get_model_variants(arch)

        return all_models


class SyntheticDataLoader:
    """Generator for synthetic inference request data and contexts."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize synthetic data loader.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.image_size = config["data"]["image_size"]
        self.batch_size = config["data"]["batch_size"]

        # Set random seeds
        random.seed(config.get("seed", 42))
        np.random.seed(config.get("numpy_seed", 42))
        torch.manual_seed(config.get("torch_seed", 42))

        logger.info("Initialized SyntheticDataLoader")

    def generate_request_context(self, batch_size: int = None) -> Dict[str, torch.Tensor]:
        """Generate synthetic request context features.

        Args:
            batch_size: Number of samples to generate. If None, uses config default.

        Returns:
            Dictionary containing context features.
        """
        if batch_size is None:
            batch_size = self.batch_size

        # Generate various context features
        context = {
            "request_complexity": torch.rand(batch_size, 1) * 100,  # 0-100 complexity score
            "historical_latency": torch.empty(batch_size, 1).exponential_(lambd=1.0) * 50,  # ms
            "gpu_memory_usage": torch.rand(batch_size, 1) * 0.9 + 0.1,  # 10-100% usage
            "queue_length": torch.randint(0, 100, (batch_size, 1)).float(),
            "model_load": torch.rand(batch_size, 1),  # 0-1 normalized load
            "time_of_day": torch.rand(batch_size, 1) * 24,  # 0-24 hour
            "user_priority": torch.randint(1, 6, (batch_size, 1)).float(),  # 1-5 priority
            "accuracy_requirement": torch.rand(batch_size, 1) * 0.1 + 0.9,  # 0.9-1.0
        }

        # Add temporal patterns
        t = time.time()
        context["request_timestamp"] = torch.tensor([t] * batch_size).unsqueeze(1)

        # Add weekly/daily patterns
        day_of_week = (t // 86400) % 7  # 0-6
        hour_of_day = (t % 86400) // 3600  # 0-23

        context["day_of_week"] = torch.tensor([day_of_week] * batch_size).unsqueeze(1)
        context["hour_of_day"] = torch.tensor([hour_of_day] * batch_size).unsqueeze(1)

        return context

    def generate_image_batch(self, batch_size: int = None) -> torch.Tensor:
        """Generate synthetic image batch for inference.

        Args:
            batch_size: Number of images to generate.

        Returns:
            Tensor of synthetic images.
        """
        if batch_size is None:
            batch_size = self.batch_size

        # Generate random images with ImageNet-like characteristics
        images = torch.rand(batch_size, 3, self.image_size, self.image_size)

        # Apply normalization similar to ImageNet
        normalize = transforms.Normalize(
            mean=self.config["data"]["normalize"]["mean"],
            std=self.config["data"]["normalize"]["std"]
        )

        images = normalize(images)
        return images

    def create_benchmark_dataset(self, num_samples: int = 10000) -> Dataset:
        """Create a benchmark dataset for evaluation.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            PyTorch Dataset for benchmarking.
        """
        return SyntheticBenchmarkDataset(
            num_samples=num_samples,
            image_size=self.image_size,
            config=self.config
        )


class SyntheticBenchmarkDataset(Dataset):
    """Dataset for synthetic benchmark data."""

    def __init__(self, num_samples: int, image_size: int, config: Dict[str, Any]) -> None:
        """Initialize dataset.

        Args:
            num_samples: Number of samples in dataset.
            image_size: Size of generated images.
            config: Configuration dictionary.
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.config = config

        # Pre-generate some data for consistency
        torch.manual_seed(config.get("torch_seed", 42))
        self.contexts = []

        loader = SyntheticDataLoader(config)
        for _ in range(0, num_samples, config["data"]["batch_size"]):
            batch_size = min(config["data"]["batch_size"],
                           num_samples - len(self.contexts))
            context = loader.generate_request_context(batch_size)
            self.contexts.extend([{k: v[i] for k, v in context.items()}
                                for i in range(batch_size)])

    def __len__(self) -> int:
        """Get dataset length."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get dataset item.

        Args:
            idx: Item index.

        Returns:
            Tuple of (image, context).
        """
        # Generate image on-the-fly to save memory
        image = torch.rand(3, self.image_size, self.image_size)

        # Apply normalization
        normalize = transforms.Normalize(
            mean=self.config["data"]["normalize"]["mean"],
            std=self.config["data"]["normalize"]["std"]
        )
        image = normalize(image)

        context = self.contexts[idx]
        return image, context


def create_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders.

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    synthetic_loader = SyntheticDataLoader(config)

    # Create datasets
    total_samples = 50000  # Default dataset size
    val_split = config["data"]["validation_split"]
    val_samples = int(total_samples * val_split)
    train_samples = total_samples - val_samples

    train_dataset = synthetic_loader.create_benchmark_dataset(train_samples)
    val_dataset = synthetic_loader.create_benchmark_dataset(val_samples)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True
    )

    logger.info(f"Created data loaders: train={len(train_loader)} batches, "
               f"val={len(val_loader)} batches")

    return train_loader, val_loader