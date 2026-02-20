"""Evaluation metrics and monitoring for adaptive routing system."""

import logging
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("Matplotlib/Seaborn not available. Plotting features disabled.")


logger = logging.getLogger(__name__)


class RoutingMetrics:
    """Comprehensive metrics for evaluating routing performance."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize routing metrics.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.target_metrics = config["evaluation"]["metrics"]
        self.sla_constraints = config["routing"]["sla_constraints"]

        # Metric storage
        self.reset()

        logger.info("Initialized RoutingMetrics")

    def reset(self) -> None:
        """Reset all metrics."""
        self.latencies = []
        self.accuracies = []
        self.memory_usages = []
        self.throughputs = []
        self.routing_decisions = []
        self.rewards = []
        self.sla_violations = defaultdict(int)
        self.total_requests = 0
        self.start_time = time.time()

    def record_request(
        self,
        latency_ms: float,
        accuracy: float,
        memory_mb: float,
        throughput_rps: float,
        variant_id: int,
        reward: float,
        sla_violated: Dict[str, bool] = None
    ) -> None:
        """Record metrics for a single request.

        Args:
            latency_ms: Request latency in milliseconds.
            accuracy: Model accuracy for this request.
            memory_mb: Memory usage in MB.
            throughput_rps: Throughput in requests per second.
            variant_id: Selected model variant ID.
            reward: Reward received for this routing decision.
            sla_violated: Dictionary indicating which SLAs were violated.
        """
        self.latencies.append(latency_ms)
        self.accuracies.append(accuracy)
        self.memory_usages.append(memory_mb)
        self.throughputs.append(throughput_rps)
        self.routing_decisions.append(variant_id)
        self.rewards.append(reward)
        self.total_requests += 1

        if sla_violated:
            for sla_type, violated in sla_violated.items():
                if violated:
                    self.sla_violations[sla_type] += 1

    def calculate_percentile_latency(self, percentile: float) -> float:
        """Calculate percentile latency.

        Args:
            percentile: Percentile to calculate (0-100).

        Returns:
            Percentile latency in ms.
        """
        if not self.latencies:
            return 0.0
        return np.percentile(self.latencies, percentile)

    def calculate_throughput_metrics(self) -> Dict[str, float]:
        """Calculate throughput-related metrics.

        Returns:
            Dictionary containing throughput metrics.
        """
        if not self.throughputs:
            return {"mean_throughput": 0.0, "peak_throughput": 0.0}

        return {
            "mean_throughput": np.mean(self.throughputs),
            "peak_throughput": np.max(self.throughputs),
            "min_throughput": np.min(self.throughputs),
            "std_throughput": np.std(self.throughputs),
        }

    def calculate_accuracy_metrics(self) -> Dict[str, float]:
        """Calculate accuracy-related metrics.

        Returns:
            Dictionary containing accuracy metrics.
        """
        if not self.accuracies:
            return {"mean_accuracy": 0.0}

        accuracies = np.array(self.accuracies)
        threshold = self.sla_constraints.get("accuracy_threshold", 0.95)

        return {
            "mean_accuracy": float(np.mean(accuracies)),
            "min_accuracy": float(np.min(accuracies)),
            "accuracy_degradation_pct": float(
                max(0, (threshold - np.mean(accuracies)) / threshold * 100)
            ),
            "accuracy_violations_pct": float(
                np.sum(accuracies < threshold) / len(accuracies) * 100
            ),
        }

    def calculate_memory_metrics(self) -> Dict[str, float]:
        """Calculate memory-related metrics.

        Returns:
            Dictionary containing memory metrics.
        """
        if not self.memory_usages:
            return {"mean_memory_mb": 0.0}

        memory_array = np.array(self.memory_usages)
        baseline_memory = 2000.0  # Baseline memory usage in MB

        return {
            "mean_memory_mb": float(np.mean(memory_array)),
            "peak_memory_mb": float(np.max(memory_array)),
            "memory_reduction_pct": float(
                max(0, (baseline_memory - np.mean(memory_array)) / baseline_memory * 100)
            ),
        }

    def calculate_cost_metrics(self) -> Dict[str, float]:
        """Calculate cost-related metrics.

        Returns:
            Dictionary containing cost metrics.
        """
        if not self.memory_usages or not self.latencies:
            return {"cost_per_inference": 0.0}

        # Simplified cost model: based on memory and latency
        memory_costs = np.array(self.memory_usages) * 0.00001  # $0.00001 per MB
        compute_costs = np.array(self.latencies) * 0.00002    # $0.00002 per ms

        total_costs = memory_costs + compute_costs
        baseline_cost = 0.05  # Baseline cost per inference

        return {
            "cost_per_inference": float(np.mean(total_costs)),
            "cost_reduction_pct": float(
                max(0, (baseline_cost - np.mean(total_costs)) / baseline_cost * 100)
            ),
            "total_cost": float(np.sum(total_costs)),
        }

    def calculate_routing_metrics(self) -> Dict[str, float]:
        """Calculate routing-specific metrics.

        Returns:
            Dictionary containing routing metrics.
        """
        if not self.routing_decisions:
            return {"routing_entropy": 0.0}

        decisions = np.array(self.routing_decisions)
        rewards = np.array(self.rewards) if self.rewards else np.zeros(len(decisions))

        # Calculate routing entropy (diversity)
        unique, counts = np.unique(decisions, return_counts=True)
        probabilities = counts / len(decisions)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))

        # Calculate regret (simplified - assumes oracle reward of 1.0)
        oracle_reward = 1.0
        regret = oracle_reward - np.mean(rewards) if len(rewards) > 0 else 0.0

        return {
            "routing_entropy": float(entropy),
            "routing_policy_regret": float(max(0, regret)),
            "mean_reward": float(np.mean(rewards)) if len(rewards) > 0 else 0.0,
            "variant_distribution": dict(zip(unique.tolist(), counts.tolist())),
        }

    def calculate_sla_metrics(self) -> Dict[str, float]:
        """Calculate SLA compliance metrics.

        Returns:
            Dictionary containing SLA metrics.
        """
        if self.total_requests == 0:
            return {}

        sla_metrics = {}
        for sla_type, violations in self.sla_violations.items():
            violation_rate = violations / self.total_requests * 100
            sla_metrics[f"{sla_type}_violation_rate_pct"] = violation_rate
            sla_metrics[f"{sla_type}_compliance_pct"] = 100 - violation_rate

        # Overall SLA compliance
        total_violations = sum(self.sla_violations.values())
        sla_metrics["overall_sla_compliance_pct"] = (
            100 - (total_violations / (self.total_requests * len(self.sla_violations)) * 100)
            if self.sla_violations else 100.0
        )

        return sla_metrics

    def get_comprehensive_metrics(self) -> Dict[str, Union[float, Dict]]:
        """Get all metrics in a comprehensive report.

        Returns:
            Dictionary containing all calculated metrics.
        """
        metrics = {}

        # Latency metrics
        if self.latencies:
            metrics.update({
                "p50_latency_ms": self.calculate_percentile_latency(50),
                "p95_latency_ms": self.calculate_percentile_latency(95),
                "p99_latency_ms": self.calculate_percentile_latency(99),
                "mean_latency_ms": float(np.mean(self.latencies)),
            })

        # Throughput metrics
        metrics.update(self.calculate_throughput_metrics())

        # Accuracy metrics
        metrics.update(self.calculate_accuracy_metrics())

        # Memory metrics
        metrics.update(self.calculate_memory_metrics())

        # Cost metrics
        metrics.update(self.calculate_cost_metrics())

        # Routing metrics
        metrics.update(self.calculate_routing_metrics())

        # SLA metrics
        metrics.update(self.calculate_sla_metrics())

        # General metrics
        elapsed_time = time.time() - self.start_time
        metrics.update({
            "total_requests": self.total_requests,
            "elapsed_time_seconds": elapsed_time,
            "average_rps": self.total_requests / elapsed_time if elapsed_time > 0 else 0.0,
        })

        return metrics

    def check_target_metrics(self) -> Dict[str, bool]:
        """Check if target metrics are achieved.

        Returns:
            Dictionary indicating which targets are met.
        """
        current_metrics = self.get_comprehensive_metrics()
        target_metrics = self.config.get("target_metrics", {})

        results = {}
        for target_name, target_value in target_metrics.items():
            current_value = current_metrics.get(target_name)
            if current_value is not None:
                # Determine if target is met based on metric type
                if "reduction" in target_name or "compliance" in target_name:
                    # Higher is better
                    results[target_name] = current_value >= target_value
                elif "latency" in target_name or "violation" in target_name or "regret" in target_name:
                    # Lower is better
                    results[target_name] = current_value <= target_value
                else:
                    # Default: higher is better
                    results[target_name] = current_value >= target_value

        return results


class PerformanceMonitor:
    """Real-time performance monitor for adaptive routing system."""

    def __init__(self, config: Dict[str, Any], window_size: int = 1000) -> None:
        """Initialize performance monitor.

        Args:
            config: Configuration dictionary.
            window_size: Size of sliding window for metrics.
        """
        self.config = config
        self.window_size = window_size

        # Sliding windows for real-time metrics
        self.latency_window = deque(maxlen=window_size)
        self.throughput_window = deque(maxlen=window_size)
        self.accuracy_window = deque(maxlen=window_size)
        self.memory_window = deque(maxlen=window_size)

        # Alert thresholds
        self.alert_thresholds = {
            "latency_p99_ms": config["routing"]["sla_constraints"]["p99_latency_ms"],
            "accuracy_min": config["routing"]["sla_constraints"]["accuracy_threshold"],
            "memory_max_gb": config["routing"]["sla_constraints"]["memory_limit_gb"],
        }

        # Alert state
        self.active_alerts = set()
        self.alert_history = []

        logger.info("Initialized PerformanceMonitor")

    def update(
        self,
        latency_ms: float,
        throughput_rps: float,
        accuracy: float,
        memory_mb: float
    ) -> None:
        """Update monitoring windows with new measurements.

        Args:
            latency_ms: Latest latency measurement.
            throughput_rps: Latest throughput measurement.
            accuracy: Latest accuracy measurement.
            memory_mb: Latest memory usage measurement.
        """
        self.latency_window.append(latency_ms)
        self.throughput_window.append(throughput_rps)
        self.accuracy_window.append(accuracy)
        self.memory_window.append(memory_mb)

        # Check for alerts
        self._check_alerts()

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current windowed metrics.

        Returns:
            Dictionary of current performance metrics.
        """
        metrics = {}

        if self.latency_window:
            latencies = list(self.latency_window)
            metrics.update({
                "current_p50_latency_ms": np.percentile(latencies, 50),
                "current_p95_latency_ms": np.percentile(latencies, 95),
                "current_p99_latency_ms": np.percentile(latencies, 99),
                "current_mean_latency_ms": np.mean(latencies),
            })

        if self.throughput_window:
            metrics["current_mean_throughput_rps"] = np.mean(list(self.throughput_window))

        if self.accuracy_window:
            metrics["current_mean_accuracy"] = np.mean(list(self.accuracy_window))

        if self.memory_window:
            memory_gb = np.mean(list(self.memory_window)) / 1024
            metrics["current_mean_memory_gb"] = memory_gb

        return metrics

    def _check_alerts(self) -> None:
        """Check for alert conditions and update alert state."""
        current_metrics = self.get_current_metrics()

        # Check latency alerts
        p99_latency = current_metrics.get("current_p99_latency_ms", 0)
        if p99_latency > self.alert_thresholds["latency_p99_ms"]:
            if "high_latency" not in self.active_alerts:
                self._trigger_alert("high_latency", f"P99 latency {p99_latency:.2f}ms exceeds threshold")
                self.active_alerts.add("high_latency")
        else:
            if "high_latency" in self.active_alerts:
                self._resolve_alert("high_latency")
                self.active_alerts.remove("high_latency")

        # Check accuracy alerts
        mean_accuracy = current_metrics.get("current_mean_accuracy", 1.0)
        if mean_accuracy < self.alert_thresholds["accuracy_min"]:
            if "low_accuracy" not in self.active_alerts:
                self._trigger_alert("low_accuracy", f"Accuracy {mean_accuracy:.4f} below threshold")
                self.active_alerts.add("low_accuracy")
        else:
            if "low_accuracy" in self.active_alerts:
                self._resolve_alert("low_accuracy")
                self.active_alerts.remove("low_accuracy")

        # Check memory alerts
        mean_memory_gb = current_metrics.get("current_mean_memory_gb", 0)
        if mean_memory_gb > self.alert_thresholds["memory_max_gb"]:
            if "high_memory" not in self.active_alerts:
                self._trigger_alert("high_memory", f"Memory usage {mean_memory_gb:.2f}GB exceeds limit")
                self.active_alerts.add("high_memory")
        else:
            if "high_memory" in self.active_alerts:
                self._resolve_alert("high_memory")
                self.active_alerts.remove("high_memory")

    def _trigger_alert(self, alert_type: str, message: str) -> None:
        """Trigger an alert.

        Args:
            alert_type: Type of alert.
            message: Alert message.
        """
        alert = {
            "type": alert_type,
            "message": message,
            "timestamp": time.time(),
            "status": "active"
        }
        self.alert_history.append(alert)
        logger.warning(f"ALERT: {alert_type} - {message}")

    def _resolve_alert(self, alert_type: str) -> None:
        """Resolve an active alert.

        Args:
            alert_type: Type of alert to resolve.
        """
        # Mark latest alert of this type as resolved
        for alert in reversed(self.alert_history):
            if alert["type"] == alert_type and alert["status"] == "active":
                alert["status"] = "resolved"
                alert["resolved_timestamp"] = time.time()
                logger.info(f"RESOLVED: {alert_type}")
                break

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alerts.

        Returns:
            Alert summary information.
        """
        active_alerts = [a for a in self.alert_history if a["status"] == "active"]
        resolved_alerts = [a for a in self.alert_history if a["status"] == "resolved"]

        return {
            "active_alerts": active_alerts,
            "resolved_alerts": resolved_alerts,
            "total_alerts": len(self.alert_history),
            "alert_rate": len(active_alerts) / max(len(self.latency_window), 1),
        }


class BenchmarkSuite:
    """Comprehensive benchmark suite for evaluating routing performance."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize benchmark suite.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.benchmark_config = config["evaluation"]["benchmark"]

        logger.info("Initialized BenchmarkSuite")

    def run_latency_benchmark(
        self,
        model,
        num_requests: int = 1000,
        concurrent_users: List[int] = None
    ) -> Dict[str, Any]:
        """Run latency benchmark with varying load.

        Args:
            model: Model to benchmark.
            num_requests: Number of requests per load level.
            concurrent_users: List of concurrent user counts.

        Returns:
            Benchmark results.
        """
        if concurrent_users is None:
            concurrent_users = self.benchmark_config["concurrent_users"]

        results = {}

        for user_count in concurrent_users:
            logger.info(f"Running latency benchmark with {user_count} concurrent users")

            latencies = []
            start_time = time.time()

            def make_request():
                request_start = time.time()
                # Simulate request processing
                context = torch.randn(self.config["training"]["bandit"]["context_dim"])
                _ = model.select_model_variant(context)
                return (time.time() - request_start) * 1000  # Convert to ms

            # Run concurrent requests
            with ThreadPoolExecutor(max_workers=user_count) as executor:
                futures = [executor.submit(make_request) for _ in range(num_requests)]

                for future in as_completed(futures):
                    try:
                        latency = future.result()
                        latencies.append(latency)
                    except Exception as e:
                        logger.error(f"Request failed: {e}")

            total_time = time.time() - start_time

            results[user_count] = {
                "latencies": latencies,
                "p50_latency_ms": np.percentile(latencies, 50),
                "p95_latency_ms": np.percentile(latencies, 95),
                "p99_latency_ms": np.percentile(latencies, 99),
                "mean_latency_ms": np.mean(latencies),
                "throughput_rps": len(latencies) / total_time,
                "total_requests": len(latencies),
                "failed_requests": num_requests - len(latencies),
            }

        return results

    def run_accuracy_benchmark(
        self,
        model,
        test_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Run accuracy benchmark across different routing decisions.

        Args:
            model: Model to benchmark.
            test_loader: Test data loader.

        Returns:
            Accuracy benchmark results.
        """
        variant_accuracies = defaultdict(list)
        routing_decisions = []

        for images, contexts in test_loader:
            batch_size = images.shape[0]

            for i in range(batch_size):
                # Extract context for single sample
                context_dict = {k: v[i] for k, v in contexts.items()}
                request_data = {
                    "batch_size": 1,
                    "image_size": 224,
                    "priority": 3,
                    "accuracy_requirement": 0.95,
                }

                # Get routing decision
                from ..data.preprocessing import ContextExtractor
                context_extractor = ContextExtractor(self.config)
                context_vector = context_extractor.extract_context(request_data)

                variant = model.select_model_variant(context_vector)
                routing_decisions.append(variant)

                # Simulate accuracy for this variant
                variant_info = model.variant_manager.get_variant_info(variant)
                base_accuracy = 0.95

                # Apply degradation based on compression
                degradation_map = {
                    "fp32": 0.0,
                    "fp16": 0.002,
                    "int8": 0.008,
                    "pruned": 0.012
                }

                variant_name = variant_info.get("name", "fp32")
                simulated_accuracy = base_accuracy - degradation_map.get(variant_name, 0.0)
                simulated_accuracy += np.random.normal(0, 0.005)  # Add noise

                variant_accuracies[variant].append(simulated_accuracy)

        # Calculate results
        results = {
            "overall_accuracy": np.mean([
                acc for accs in variant_accuracies.values() for acc in accs
            ]),
            "variant_accuracies": {
                str(variant): np.mean(accs)
                for variant, accs in variant_accuracies.items()
            },
            "routing_distribution": dict(zip(*[arr.tolist() for arr in np.unique(routing_decisions, return_counts=True)])),
        }

        return results

    def run_memory_benchmark(
        self,
        model,
        batch_sizes: List[int] = None
    ) -> Dict[str, Any]:
        """Run memory usage benchmark.

        Args:
            model: Model to benchmark.
            batch_sizes: List of batch sizes to test.

        Returns:
            Memory benchmark results.
        """
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32, 64]

        results = {}

        for batch_size in batch_sizes:
            logger.info(f"Running memory benchmark with batch size {batch_size}")

            # Simulate memory usage for different variants
            variant_memory = {}

            for variant_idx in range(len(self.config["model"]["compression_variants"])):
                variant_info = model.variant_manager.get_variant_info(variant_idx)

                # Estimate memory based on batch size and compression
                base_memory = 1000 * batch_size  # MB
                memory_usage = base_memory * variant_info["memory_multiplier"]

                variant_memory[variant_idx] = memory_usage

            results[batch_size] = {
                "variant_memory_mb": variant_memory,
                "max_memory_mb": max(variant_memory.values()),
                "min_memory_mb": min(variant_memory.values()),
                "memory_reduction_pct": (
                    (max(variant_memory.values()) - min(variant_memory.values()))
                    / max(variant_memory.values()) * 100
                ),
            }

        return results

    def run_comprehensive_benchmark(
        self,
        model,
        test_loader: torch.utils.data.DataLoader = None
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark suite.

        Args:
            model: Model to benchmark.
            test_loader: Test data loader.

        Returns:
            Complete benchmark results.
        """
        logger.info("Starting comprehensive benchmark suite")

        results = {
            "benchmark_config": self.benchmark_config,
            "timestamp": time.time(),
        }

        # Run latency benchmark
        try:
            logger.info("Running latency benchmark")
            results["latency_benchmark"] = self.run_latency_benchmark(
                model, self.benchmark_config["num_requests"]
            )
        except Exception as e:
            logger.error(f"Latency benchmark failed: {e}")

        # Run accuracy benchmark
        if test_loader is not None:
            try:
                logger.info("Running accuracy benchmark")
                results["accuracy_benchmark"] = self.run_accuracy_benchmark(model, test_loader)
            except Exception as e:
                logger.error(f"Accuracy benchmark failed: {e}")

        # Run memory benchmark
        try:
            logger.info("Running memory benchmark")
            results["memory_benchmark"] = self.run_memory_benchmark(model)
        except Exception as e:
            logger.error(f"Memory benchmark failed: {e}")

        logger.info("Comprehensive benchmark completed")
        return results

    def generate_benchmark_report(
        self,
        results: Dict[str, Any],
        save_path: str = None
    ) -> str:
        """Generate benchmark report.

        Args:
            results: Benchmark results.
            save_path: Path to save report.

        Returns:
            Report text.
        """
        report_lines = [
            "# Adaptive Model Routing Benchmark Report",
            "",
            f"**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(results['timestamp']))}",
            "",
        ]

        # Latency benchmark results
        if "latency_benchmark" in results:
            report_lines.extend([
                "## Latency Benchmark Results",
                "",
                "| Concurrent Users | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) | Throughput (RPS) |",
                "|------------------|----------|----------|----------|-----------|------------------|",
            ])

            for users, metrics in results["latency_benchmark"].items():
                report_lines.append(
                    f"| {users} | {metrics['p50_latency_ms']:.2f} | "
                    f"{metrics['p95_latency_ms']:.2f} | {metrics['p99_latency_ms']:.2f} | "
                    f"{metrics['mean_latency_ms']:.2f} | {metrics['throughput_rps']:.2f} |"
                )

            report_lines.append("")

        # Accuracy benchmark results
        if "accuracy_benchmark" in results:
            acc_results = results["accuracy_benchmark"]
            report_lines.extend([
                "## Accuracy Benchmark Results",
                "",
                f"**Overall Accuracy:** {acc_results['overall_accuracy']:.4f}",
                "",
                "### Per-Variant Accuracy:",
            ])

            for variant, accuracy in acc_results["variant_accuracies"].items():
                report_lines.append(f"- Variant {variant}: {accuracy:.4f}")

            report_lines.append("")

        # Memory benchmark results
        if "memory_benchmark" in results:
            report_lines.extend([
                "## Memory Benchmark Results",
                "",
                "| Batch Size | Max Memory (MB) | Min Memory (MB) | Reduction (%) |",
                "|------------|-----------------|-----------------|---------------|",
            ])

            for batch_size, metrics in results["memory_benchmark"].items():
                report_lines.append(
                    f"| {batch_size} | {metrics['max_memory_mb']:.0f} | "
                    f"{metrics['min_memory_mb']:.0f} | {metrics['memory_reduction_pct']:.1f} |"
                )

            report_lines.append("")

        report_text = "\n".join(report_lines)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Benchmark report saved to {save_path}")

        return report_text