#!/usr/bin/env python3
"""Simple test script to identify import and basic functionality issues."""

import sys
import os
import traceback
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test all imports to identify missing dependencies or syntax errors."""
    print("Testing imports...")

    try:
        from adaptive_model_routing_serving_optimizer.utils.config import Config, load_config
        print("✓ Config imports successful")
    except Exception as e:
        print(f"✗ Config import failed: {e}")
        traceback.print_exc()

    try:
        from adaptive_model_routing_serving_optimizer.data.loader import (
            ModelZooLoader, SyntheticDataLoader, create_data_loaders
        )
        print("✓ Data loader imports successful")
    except Exception as e:
        print(f"✗ Data loader import failed: {e}")
        traceback.print_exc()

    try:
        from adaptive_model_routing_serving_optimizer.data.preprocessing import (
            RequestPreprocessor, ContextExtractor, SystemMonitor
        )
        print("✓ Preprocessing imports successful")
    except Exception as e:
        print(f"✗ Preprocessing import failed: {e}")
        traceback.print_exc()

    try:
        from adaptive_model_routing_serving_optimizer.models.model import (
            AdaptiveRoutingModel, ContextualBandit, RoutingPolicy, ModelVariantManager
        )
        print("✓ Model imports successful")
    except Exception as e:
        print(f"✗ Model import failed: {e}")
        traceback.print_exc()

    try:
        from adaptive_model_routing_serving_optimizer.training.trainer import (
            RoutingTrainer, OnlineTrainer
        )
        print("✓ Training imports successful")
    except Exception as e:
        print(f"✗ Training import failed: {e}")
        traceback.print_exc()

    try:
        from adaptive_model_routing_serving_optimizer.evaluation.metrics import (
            RoutingMetrics, PerformanceMonitor, BenchmarkSuite
        )
        print("✓ Evaluation imports successful")
    except Exception as e:
        print(f"✗ Evaluation import failed: {e}")
        traceback.print_exc()

def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")

    try:
        # Test config loading
        config = load_config("configs/default.yaml")
        print("✓ Config loading successful")

        # Test model creation
        from adaptive_model_routing_serving_optimizer.models.model import AdaptiveRoutingModel
        model = AdaptiveRoutingModel(config._config)
        print("✓ Model creation successful")

        # Test data loader creation
        from adaptive_model_routing_serving_optimizer.data.loader import create_data_loaders
        train_loader, val_loader = create_data_loaders(config._config)
        print(f"✓ Data loader creation successful ({len(train_loader)} train batches, {len(val_loader)} val batches)")

    except Exception as e:
        print(f"✗ Basic functionality failed: {e}")
        traceback.print_exc()

def test_gpu_availability():
    """Test GPU availability and CUDA functionality."""
    print("\nTesting GPU availability...")

    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("ℹ No GPU available, will use CPU")
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
        traceback.print_exc()

def test_training_script():
    """Test that training script can be executed."""
    print("\nTesting training script...")

    try:
        # Import training script
        sys.path.append("scripts")

        # Check if config exists
        config_path = Path("configs/default.yaml")
        if config_path.exists():
            print("✓ Config file exists")
        else:
            print("✗ Config file missing")

        # Check if checkpoint directories can be created
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        print("✓ Checkpoint directory created")

        # Check if logs directory can be created
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        print("✓ Logs directory created")

        # Check if results directory can be created
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        print("✓ Results directory created")

    except Exception as e:
        print(f"✗ Training script test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 60)
    print("ADAPTIVE MODEL ROUTING - DIAGNOSTIC TEST")
    print("=" * 60)

    test_imports()
    test_gpu_availability()
    test_basic_functionality()
    test_training_script()

    print("\n" + "=" * 60)
    print("DIAGNOSTIC TEST COMPLETED")
    print("=" * 60)