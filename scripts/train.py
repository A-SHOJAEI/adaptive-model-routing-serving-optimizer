#!/usr/bin/env python3
"""Training script for adaptive model routing serving optimizer.

This script trains the adaptive routing model using contextual bandits and
policy networks to learn optimal model variant routing decisions.
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_model_routing_serving_optimizer.utils.config import load_config
from adaptive_model_routing_serving_optimizer.models.model import AdaptiveRoutingModel
from adaptive_model_routing_serving_optimizer.training.trainer import RoutingTrainer
from adaptive_model_routing_serving_optimizer.data.loader import create_data_loaders
from adaptive_model_routing_serving_optimizer.evaluation.metrics import (
    RoutingMetrics,
    BenchmarkSuite
)


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper()))
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    root.handlers.clear()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    root.addHandler(sh)
    fh = logging.FileHandler("logs/training.log", mode="w")
    fh.setFormatter(formatter)
    root.addHandler(fh)


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set environment variable for Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train adaptive model routing serving optimizer"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for training (cuda/cpu)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only run evaluation without training"
    )

    args = parser.parse_args()

    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path(args.checkpoint_dir).mkdir(exist_ok=True)

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting adaptive model routing training")

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Set random seeds
        seed = config.get("seed", 42)
        set_random_seeds(seed)
        logger.info(f"Set random seed to {seed}")

        # Determine device
        if args.device:
            device = torch.device(args.device)
        else:
            device = torch.device(
                config.get("hardware", {}).get("device", "cuda")
                if torch.cuda.is_available() else "cpu"
            )
        logger.info(f"Using device: {device}")

        # Create model
        logger.info("Initializing adaptive routing model")
        model = AdaptiveRoutingModel(config._config)
        model.to(device)

        # Log model information
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {num_params:,} total, {num_trainable:,} trainable")

        # Create data loaders
        logger.info("Creating data loaders")
        train_loader, val_loader = create_data_loaders(config._config)
        logger.info(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

        if not args.evaluate_only:
            # Create trainer
            logger.info("Initializing trainer")
            trainer = RoutingTrainer(model, config._config, device)

            # Resume from checkpoint if specified
            if args.resume_from:
                logger.info(f"Resuming training from {args.resume_from}")
                trainer.load_checkpoint(args.resume_from)

            # Train model
            logger.info("Starting training")
            training_history = trainer.train(
                train_loader,
                val_loader,
                save_dir=args.checkpoint_dir
            )

            # Save training results
            results_path = Path("results") / "training_history.json"
            import json
            with open(results_path, "w") as f:
                json.dump(training_history, f, indent=2, default=str)
            logger.info(f"Training history saved to {results_path}")

        else:
            logger.info("Evaluation-only mode: loading best checkpoint")
            best_checkpoint = Path(args.checkpoint_dir) / "best_model.pth"
            if best_checkpoint.exists():
                checkpoint = torch.load(best_checkpoint, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                logger.info("Loaded best model checkpoint")
            else:
                logger.warning("No best checkpoint found, using initialized model")

        # Run evaluation
        logger.info("Running comprehensive evaluation")
        model.eval()

        # Initialize metrics
        routing_metrics = RoutingMetrics(config._config)
        benchmark_suite = BenchmarkSuite(config._config)

        # Simulate some routing decisions for evaluation
        logger.info("Simulating routing decisions for evaluation")
        num_eval_samples = 1000

        with torch.no_grad():
            for i in range(num_eval_samples):
                # Generate synthetic context
                context = torch.randn(config.get_nested("training", "bandit", "context_dim")).to(device)

                # Get routing decision
                variant = model.select_model_variant(context, use_bandit=False)  # Use policy for evaluation

                # Simulate performance
                request_data = {"batch_size": 1, "image_size": 224, "priority": 3}
                simulated_performance = model.variant_manager.estimate_performance(variant, request_data)

                # Add realistic noise
                simulated_performance["latency_ms"] += np.random.normal(0, 5)
                simulated_performance["accuracy"] += np.random.normal(0, 0.005)
                simulated_performance["memory_mb"] += np.random.normal(0, 50)
                simulated_performance["throughput_rps"] = max(1, simulated_performance["throughput_rps"])

                # Check SLA violations
                sla_constraints = config.get_nested("routing", "sla_constraints")
                sla_violated = {
                    "latency": simulated_performance["latency_ms"] > sla_constraints["p99_latency_ms"],
                    "accuracy": simulated_performance["accuracy"] < sla_constraints["accuracy_threshold"],
                    "memory": simulated_performance["memory_mb"] > sla_constraints["memory_limit_gb"] * 1024
                }

                # Calculate reward
                reward = model.variant_manager.calculate_reward(
                    variant, simulated_performance, sla_constraints
                )

                # Record metrics
                routing_metrics.record_request(
                    latency_ms=simulated_performance["latency_ms"],
                    accuracy=simulated_performance["accuracy"],
                    memory_mb=simulated_performance["memory_mb"],
                    throughput_rps=simulated_performance["throughput_rps"],
                    variant_id=variant,
                    reward=reward,
                    sla_violated=sla_violated
                )

        # Calculate final metrics
        final_metrics = routing_metrics.get_comprehensive_metrics()
        logger.info("Evaluation metrics:")
        for key, value in final_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}")

        # Check target achievements
        target_check = routing_metrics.check_target_metrics()
        logger.info("Target metric achievements:")
        for metric, achieved in target_check.items():
            status = "✓" if achieved else "✗"
            logger.info(f"  {status} {metric}: {achieved}")

        # Run benchmark suite
        logger.info("Running benchmark suite")
        try:
            benchmark_results = benchmark_suite.run_comprehensive_benchmark(model, val_loader)

            # Save benchmark results
            benchmark_path = Path("results") / "benchmark_results.json"
            import json
            with open(benchmark_path, "w") as f:
                json.dump(benchmark_results, f, indent=2, default=str)
            logger.info(f"Benchmark results saved to {benchmark_path}")

            # Generate benchmark report
            report_path = Path("results") / "benchmark_report.md"
            report = benchmark_suite.generate_benchmark_report(
                benchmark_results, str(report_path)
            )
            logger.info(f"Benchmark report saved to {report_path}")

        except Exception as e:
            logger.warning(f"Benchmark suite failed: {e}")

        # Save final metrics
        metrics_path = Path("results") / "final_metrics.json"
        import json
        with open(metrics_path, "w") as f:
            json.dump(final_metrics, f, indent=2, default=str)
        logger.info(f"Final metrics saved to {metrics_path}")

        # Print summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Device: {device}")
        print(f"Total parameters: {num_params:,}")
        print(f"P99 Latency: {final_metrics.get('p99_latency_ms', 0):.2f} ms")
        print(f"Mean Throughput: {final_metrics.get('mean_throughput', 0):.2f} RPS")
        print(f"Mean Accuracy: {final_metrics.get('mean_accuracy', 0):.4f}")
        print(f"Memory Reduction: {final_metrics.get('memory_reduction_pct', 0):.1f}%")
        print(f"SLA Compliance: {final_metrics.get('overall_sla_compliance_pct', 0):.1f}%")
        print(f"Routing Regret: {final_metrics.get('routing_policy_regret', 0):.4f}")

        targets_met = sum(target_check.values())
        total_targets = len(target_check)
        print(f"Targets achieved: {targets_met}/{total_targets}")
        print("="*60)

        logger.info("Training and evaluation completed successfully")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()