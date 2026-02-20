#!/usr/bin/env python3
"""Evaluation script for adaptive model routing serving optimizer.

This script evaluates trained models and generates comprehensive performance reports.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_model_routing_serving_optimizer.utils.config import load_config
from adaptive_model_routing_serving_optimizer.models.model import AdaptiveRoutingModel
from adaptive_model_routing_serving_optimizer.data.loader import create_data_loaders
from adaptive_model_routing_serving_optimizer.evaluation.metrics import (
    RoutingMetrics,
    PerformanceMonitor,
    BenchmarkSuite
)


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def load_model(checkpoint_path: str, config: Dict[str, Any], device: torch.device) -> AdaptiveRoutingModel:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        config: Configuration dictionary.
        device: Device to load model on.

    Returns:
        Loaded model.
    """
    model = AdaptiveRoutingModel(config)
    model.to(device)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logging.info(f"Loaded model from {checkpoint_path}")
    else:
        logging.warning("No checkpoint provided, using randomly initialized model")

    return model


def evaluate_routing_performance(
    model: AdaptiveRoutingModel,
    config: Dict[str, Any],
    num_samples: int = 5000
) -> Dict[str, Any]:
    """Evaluate routing performance with synthetic data.

    Args:
        model: Model to evaluate.
        config: Configuration dictionary.
        num_samples: Number of samples to evaluate.

    Returns:
        Performance metrics.
    """
    logging.info(f"Evaluating routing performance with {num_samples} samples")

    metrics = RoutingMetrics(config)
    model.eval()

    with torch.no_grad():
        for i in range(num_samples):
            # Generate context
            context_dim = config["training"]["bandit"]["context_dim"]
            context = torch.randn(context_dim)

            # Get routing decisions from both bandit and policy
            bandit_variant = model.select_model_variant(context, use_bandit=True)
            policy_variant = model.select_model_variant(context, use_bandit=False)

            # Use policy variant for evaluation (more stable)
            variant = policy_variant

            # Simulate request and performance
            request_data = {
                "batch_size": np.random.randint(1, 8),
                "image_size": np.random.choice([224, 256, 512]),
                "priority": np.random.randint(1, 6)
            }

            performance = model.variant_manager.estimate_performance(variant, request_data)

            # Add realistic noise and variability
            latency_noise = np.random.normal(0, performance["latency_ms"] * 0.1)
            accuracy_noise = np.random.normal(0, 0.01)
            memory_noise = np.random.normal(0, performance["memory_mb"] * 0.05)

            performance["latency_ms"] = max(1, performance["latency_ms"] + latency_noise)
            performance["accuracy"] = np.clip(performance["accuracy"] + accuracy_noise, 0, 1)
            performance["memory_mb"] = max(100, performance["memory_mb"] + memory_noise)

            # Recalculate throughput
            performance["throughput_rps"] = 1000 / performance["latency_ms"]

            # Check SLA violations
            sla_constraints = config["routing"]["sla_constraints"]
            sla_violated = {
                "latency": performance["latency_ms"] > sla_constraints["p99_latency_ms"],
                "accuracy": performance["accuracy"] < sla_constraints["accuracy_threshold"],
                "memory": performance["memory_mb"] > sla_constraints["memory_limit_gb"] * 1024
            }

            # Calculate reward
            reward = model.variant_manager.calculate_reward(
                variant, performance, sla_constraints
            )

            # Record metrics
            metrics.record_request(
                latency_ms=performance["latency_ms"],
                accuracy=performance["accuracy"],
                memory_mb=performance["memory_mb"],
                throughput_rps=performance["throughput_rps"],
                variant_id=variant,
                reward=reward,
                sla_violated=sla_violated
            )

    return metrics.get_comprehensive_metrics()


def compare_routing_strategies(
    model: AdaptiveRoutingModel,
    config: Dict[str, Any],
    num_samples: int = 1000
) -> Dict[str, Dict[str, Any]]:
    """Compare different routing strategies.

    Args:
        model: Model to evaluate.
        config: Configuration dictionary.
        num_samples: Number of samples for comparison.

    Returns:
        Comparison results.
    """
    logging.info("Comparing routing strategies")

    strategies = {
        "adaptive_bandit": lambda ctx: model.select_model_variant(ctx, use_bandit=True),
        "adaptive_policy": lambda ctx: model.select_model_variant(ctx, use_bandit=False),
        "random": lambda ctx: np.random.randint(0, model.num_variants),
        "always_fp32": lambda ctx: 0,  # Assuming FP32 is variant 0
        "always_fastest": lambda ctx: np.argmin([
            v["latency_multiplier"] for v in config["model"]["compression_variants"]
        ])
    }

    results = {}

    for strategy_name, strategy_fn in strategies.items():
        logging.info(f"Evaluating strategy: {strategy_name}")
        strategy_metrics = RoutingMetrics(config)

        with torch.no_grad():
            for _ in range(num_samples):
                context = torch.randn(config["training"]["bandit"]["context_dim"])
                variant = strategy_fn(context)

                # Ensure valid variant
                variant = max(0, min(variant, model.num_variants - 1))

                # Simulate performance
                request_data = {"batch_size": 1, "image_size": 224, "priority": 3}
                performance = model.variant_manager.estimate_performance(variant, request_data)

                # Add noise
                performance["latency_ms"] += np.random.normal(0, 3)
                performance["accuracy"] += np.random.normal(0, 0.003)

                sla_constraints = config["routing"]["sla_constraints"]
                reward = model.variant_manager.calculate_reward(
                    variant, performance, sla_constraints
                )

                strategy_metrics.record_request(
                    latency_ms=performance["latency_ms"],
                    accuracy=performance["accuracy"],
                    memory_mb=performance["memory_mb"],
                    throughput_rps=performance["throughput_rps"],
                    variant_id=variant,
                    reward=reward
                )

        results[strategy_name] = strategy_metrics.get_comprehensive_metrics()

    return results


def run_ablation_studies(
    model: AdaptiveRoutingModel,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run ablation studies on model components.

    Args:
        model: Model to analyze.
        config: Configuration dictionary.

    Returns:
        Ablation study results.
    """
    logging.info("Running ablation studies")

    results = {}

    # Study 1: Context feature importance
    context_dim = config["training"]["bandit"]["context_dim"]
    base_context = torch.randn(context_dim)

    feature_importance = []
    for i in range(context_dim):
        # Ablate feature i
        modified_context = base_context.clone()
        modified_context[i] = 0

        base_variant = model.select_model_variant(base_context, use_bandit=False)
        modified_variant = model.select_model_variant(modified_context, use_bandit=False)

        # Measure change in decision
        decision_change = int(base_variant != modified_variant)
        feature_importance.append(decision_change)

    results["feature_importance"] = feature_importance

    # Study 2: Exploration parameter sensitivity
    original_exploration = model.contextual_bandit.exploration_param
    exploration_values = [0.01, 0.05, 0.1, 0.2, 0.5]
    exploration_results = {}

    for exp_val in exploration_values:
        model.contextual_bandit.exploration_param = exp_val
        decisions = []

        for _ in range(100):
            context = torch.randn(context_dim)
            variant = model.select_model_variant(context, use_bandit=True)
            decisions.append(variant)

        # Calculate decision entropy
        unique, counts = np.unique(decisions, return_counts=True)
        probs = counts / len(decisions)
        entropy = -np.sum(probs * np.log(probs + 1e-8))

        exploration_results[exp_val] = {
            "entropy": float(entropy),
            "unique_variants": len(unique),
            "most_common_variant": int(unique[np.argmax(counts)])
        }

    # Restore original exploration parameter
    model.contextual_bandit.exploration_param = original_exploration
    results["exploration_sensitivity"] = exploration_results

    # Study 3: Reward component analysis
    sample_performance = {
        "latency_ms": 45.0,
        "accuracy": 0.96,
        "memory_mb": 800.0,
        "throughput_rps": 100.0
    }

    reward_weights = config["routing"]["reward_weights"]
    component_rewards = {}

    for component in reward_weights.keys():
        # Calculate reward with only this component
        temp_weights = {k: 0.0 for k in reward_weights.keys()}
        temp_weights[component] = 1.0

        temp_config = config.copy()
        temp_config["routing"]["reward_weights"] = temp_weights

        temp_manager = model.variant_manager.__class__(temp_config)
        component_reward = temp_manager.calculate_reward(
            0, sample_performance, config["routing"]["sla_constraints"]
        )

        component_rewards[component] = float(component_reward)

    results["reward_components"] = component_rewards

    return results


def generate_evaluation_report(
    performance_metrics: Dict[str, Any],
    comparison_results: Dict[str, Dict[str, Any]],
    ablation_results: Dict[str, Any],
    config: Dict[str, Any]
) -> str:
    """Generate comprehensive evaluation report.

    Args:
        performance_metrics: Performance evaluation results.
        comparison_results: Strategy comparison results.
        ablation_results: Ablation study results.
        config: Configuration dictionary.

    Returns:
        Formatted report string.
    """
    report_lines = [
        "# Adaptive Model Routing Evaluation Report",
        "",
        "## Performance Summary",
        "",
        f"**P99 Latency**: {performance_metrics.get('p99_latency_ms', 0):.2f} ms",
        f"**Mean Throughput**: {performance_metrics.get('mean_throughput', 0):.2f} RPS",
        f"**Mean Accuracy**: {performance_metrics.get('mean_accuracy', 0):.4f}",
        f"**Memory Reduction**: {performance_metrics.get('memory_reduction_pct', 0):.1f}%",
        f"**Cost Reduction**: {performance_metrics.get('cost_reduction_pct', 0):.1f}%",
        f"**SLA Compliance**: {performance_metrics.get('overall_sla_compliance_pct', 0):.1f}%",
        "",
        "## Strategy Comparison",
        "",
        "| Strategy | P99 Latency (ms) | Mean Accuracy | Mean Reward | SLA Compliance (%) |",
        "|----------|------------------|---------------|-------------|--------------------|",
    ]

    for strategy, metrics in comparison_results.items():
        report_lines.append(
            f"| {strategy} | {metrics.get('p99_latency_ms', 0):.2f} | "
            f"{metrics.get('mean_accuracy', 0):.4f} | {metrics.get('mean_reward', 0):.3f} | "
            f"{metrics.get('overall_sla_compliance_pct', 0):.1f} |"
        )

    report_lines.extend([
        "",
        "## Ablation Studies",
        "",
        "### Feature Importance",
        "Feature importance scores (higher = more influential):",
        "",
    ])

    feature_importance = ablation_results.get("feature_importance", [])
    for i, importance in enumerate(feature_importance):
        report_lines.append(f"- Feature {i}: {importance}")

    report_lines.extend([
        "",
        "### Exploration Sensitivity",
        "",
        "| Exploration Parameter | Decision Entropy | Unique Variants |",
        "|----------------------|------------------|-----------------|",
    ])

    exploration_results = ablation_results.get("exploration_sensitivity", {})
    for exp_param, results in exploration_results.items():
        report_lines.append(
            f"| {exp_param} | {results['entropy']:.3f} | {results['unique_variants']} |"
        )

    report_lines.extend([
        "",
        "### Reward Component Analysis",
        "",
    ])

    reward_components = ablation_results.get("reward_components", {})
    for component, reward in reward_components.items():
        report_lines.append(f"- {component}: {reward:.3f}")

    return "\n".join(report_lines)


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate adaptive model routing serving optimizer"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5000,
        help="Number of samples for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    Path(args.output_dir).mkdir(exist_ok=True)

    try:
        # Load configuration
        config = load_config(args.config)

        # Determine device
        device = torch.device(
            args.device or
            config.get("hardware", {}).get("device", "cuda")
            if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using device: {device}")

        # Load model
        model = load_model(args.checkpoint, config._config, device)

        # Run performance evaluation
        logger.info("Running performance evaluation")
        performance_metrics = evaluate_routing_performance(
            model, config._config, args.num_samples
        )

        # Run strategy comparison
        logger.info("Running strategy comparison")
        comparison_results = compare_routing_strategies(
            model, config._config, min(1000, args.num_samples)
        )

        # Run ablation studies
        logger.info("Running ablation studies")
        ablation_results = run_ablation_studies(model, config._config)

        # Save results
        results = {
            "performance_metrics": performance_metrics,
            "comparison_results": comparison_results,
            "ablation_results": ablation_results,
            "config": config._config
        }

        results_path = Path(args.output_dir) / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {results_path}")

        # Generate report
        report = generate_evaluation_report(
            performance_metrics, comparison_results, ablation_results, config._config
        )

        report_path = Path(args.output_dir) / "evaluation_report.md"
        with open(report_path, "w") as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")

        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Samples evaluated: {args.num_samples:,}")
        print(f"P99 Latency: {performance_metrics.get('p99_latency_ms', 0):.2f} ms")
        print(f"Mean Accuracy: {performance_metrics.get('mean_accuracy', 0):.4f}")
        print(f"SLA Compliance: {performance_metrics.get('overall_sla_compliance_pct', 0):.1f}%")

        # Best strategy
        best_strategy = max(
            comparison_results.items(),
            key=lambda x: x[1].get('mean_reward', 0)
        )[0]
        print(f"Best strategy: {best_strategy}")
        print("="*60)

        logger.info("Evaluation completed successfully")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()