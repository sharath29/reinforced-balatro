"""
Performance Analysis Suite
==========================

Comprehensive performance metrics, benchmarking, and analysis tools.

Features:
- Algorithm comparison metrics
- Training throughput analysis
- Convergence speed metrics
- Resource utilization tracking
- Automated performance reports
"""

from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""

    algorithm: str
    total_timesteps: int
    training_duration: float  # seconds

    # Throughput metrics
    steps_per_second: float
    steps_per_minute: float
    steps_per_hour: float

    # Convergence metrics
    time_to_50pct: Optional[float] = None  # seconds to reach 50% of final reward
    time_to_75pct: Optional[float] = None
    time_to_90pct: Optional[float] = None

    # Quality metrics
    mean_reward: float = 0.0
    std_reward: float = 0.0
    max_reward: float = 0.0

    # Game-specific metrics
    mean_antes: float = 0.0
    max_antes: int = 0
    success_rate: float = 0.0  # % episodes reaching ante >= 5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'algorithm': self.algorithm,
            'total_timesteps': self.total_timesteps,
            'training_duration': self.training_duration,
            'steps_per_second': self.steps_per_second,
            'steps_per_minute': self.steps_per_minute,
            'steps_per_hour': self.steps_per_hour,
            'time_to_50pct': self.time_to_50pct,
            'time_to_75pct': self.time_to_75pct,
            'time_to_90pct': self.time_to_90pct,
            'mean_reward': self.mean_reward,
            'std_reward': self.std_reward,
            'max_reward': self.max_reward,
            'mean_antes': self.mean_antes,
            'max_antes': self.max_antes,
            'success_rate': self.success_rate,
        }

    def __str__(self) -> str:
        """String representation"""
        lines = [
            f"Performance Metrics: {self.algorithm}",
            "=" * 60,
            f"Training Duration: {self.training_duration:.1f}s",
            f"Total Timesteps: {self.total_timesteps:,}",
            f"Throughput:",
            f"  • {self.steps_per_second:.0f} steps/second",
            f"  • {self.steps_per_minute:.0f} steps/minute",
            f"  • {self.steps_per_hour:.0f} steps/hour",
        ]

        if self.time_to_50pct:
            lines.extend([
                f"Convergence Speed:",
                f"  • 50% reward in {self.time_to_50pct:.1f}s",
                f"  • 75% reward in {self.time_to_75pct:.1f}s" if self.time_to_75pct else "",
                f"  • 90% reward in {self.time_to_90pct:.1f}s" if self.time_to_90pct else "",
            ])

        lines.extend([
            f"Reward Quality:",
            f"  • Mean: {self.mean_reward:.2f} ± {self.std_reward:.2f}",
            f"  • Max: {self.max_reward:.2f}",
            f"Game Performance:",
            f"  • Mean antes: {self.mean_antes:.1f}",
            f"  • Max antes: {self.max_antes}",
            f"  • Success rate (ante≥5): {self.success_rate:.1f}%",
            "=" * 60,
        ])

        return "\n".join([line for line in lines if line])


class PerformanceAnalyzer:
    """Analyze and compare training performance"""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize analyzer

        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = Path(output_dir or "performance_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[PerformanceMetrics] = []
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def add_result(self, result: PerformanceMetrics):
        """Add a performance result"""
        self.results.append(result)

    def add_results(self, results: List[PerformanceMetrics]):
        """Add multiple results"""
        self.results.extend(results)

    def compute_throughput(self, timesteps: int, duration: float) -> PerformanceMetrics:
        """
        Compute throughput metrics

        Args:
            timesteps: Number of training steps
            duration: Training duration in seconds

        Returns:
            PerformanceMetrics with throughput data
        """
        steps_per_second = timesteps / duration if duration > 0 else 0
        steps_per_minute = steps_per_second * 60
        steps_per_hour = steps_per_second * 3600

        return PerformanceMetrics(
            algorithm="unknown",
            total_timesteps=timesteps,
            training_duration=duration,
            steps_per_second=steps_per_second,
            steps_per_minute=steps_per_minute,
            steps_per_hour=steps_per_hour,
        )

    def compute_convergence_speed(self,
                                  rewards: List[float],
                                  times: List[float]) -> Tuple[float, float, float]:
        """
        Compute convergence speed metrics

        Args:
            rewards: List of episode rewards
            times: List of times (seconds)

        Returns:
            Tuple of (time_to_50%, time_to_75%, time_to_90%)
        """
        if not rewards or not times:
            return None, None, None

        max_reward = np.max(rewards)
        rewards_array = np.array(rewards)

        time_to_50 = None
        time_to_75 = None
        time_to_90 = None

        # Find times where reward thresholds are reached
        threshold_50 = max_reward * 0.5
        threshold_75 = max_reward * 0.75
        threshold_90 = max_reward * 0.9

        for i, (reward, time) in enumerate(zip(rewards, times)):
            if time_to_50 is None and reward >= threshold_50:
                time_to_50 = time
            if time_to_75 is None and reward >= threshold_75:
                time_to_75 = time
            if time_to_90 is None and reward >= threshold_90:
                time_to_90 = time

        return time_to_50, time_to_75, time_to_90

    def generate_comparison_table(self) -> str:
        """Generate comparison table of all results"""
        if not self.results:
            return "No results to compare"

        lines = [
            "\n" + "=" * 120,
            "Performance Comparison",
            "=" * 120,
            f"{'Algorithm':<15} {'Steps/Sec':<15} {'Training Time':<15} "
            f"{'Mean Reward':<15} {'Success Rate':<15} {'Max Antes':<10}",
            "-" * 120,
        ]

        for result in sorted(self.results, key=lambda r: r.steps_per_second, reverse=True):
            algo = result.algorithm[:15]
            sps = f"{result.steps_per_second:.0f}"
            time_str = f"{result.training_duration:.1f}s"
            reward = f"{result.mean_reward:.2f}"
            success = f"{result.success_rate:.1f}%"
            antes = f"{result.max_antes}"

            lines.append(f"{algo:<15} {sps:<15} {time_str:<15} "
                        f"{reward:<15} {success:<15} {antes:<10}")

        lines.append("=" * 120 + "\n")
        return "\n".join(lines)

    def get_throughput_ranking(self) -> List[Tuple[str, float]]:
        """Get algorithms ranked by throughput"""
        ranking = [
            (r.algorithm, r.steps_per_second)
            for r in sorted(self.results, key=lambda r: r.steps_per_second, reverse=True)
        ]
        return ranking

    def get_convergence_ranking(self) -> List[Tuple[str, float]]:
        """Get algorithms ranked by convergence speed (time to 90%)"""
        ranking = []
        for r in self.results:
            if r.time_to_90pct:
                ranking.append((r.algorithm, r.time_to_90pct))

        return sorted(ranking, key=lambda x: x[1])

    def get_quality_ranking(self) -> List[Tuple[str, float]]:
        """Get algorithms ranked by reward quality"""
        ranking = [
            (r.algorithm, r.mean_reward)
            for r in sorted(self.results, key=lambda r: r.mean_reward, reverse=True)
        ]
        return ranking

    def print_summary(self):
        """Print summary of all results"""
        print(self.generate_comparison_table())

        print("Rankings by Throughput:")
        for i, (algo, sps) in enumerate(self.get_throughput_ranking(), 1):
            print(f"  {i}. {algo}: {sps:.0f} steps/sec")

        print("\nRankings by Convergence Speed:")
        for i, (algo, time) in enumerate(self.get_convergence_ranking(), 1):
            print(f"  {i}. {algo}: {time:.1f}s to 90%")

        print("\nRankings by Reward Quality:")
        for i, (algo, reward) in enumerate(self.get_quality_ranking(), 1):
            print(f"  {i}. {algo}: {reward:.2f} mean reward")

    def export_json(self, filename: Optional[str] = None) -> Path:
        """Export results to JSON"""
        filename = filename or f"performance_{self.timestamp}.json"
        filepath = self.output_dir / filename

        data = {
            'timestamp': self.timestamp,
            'results': [r.to_dict() for r in self.results],
            'rankings': {
                'throughput': self.get_throughput_ranking(),
                'convergence': self.get_convergence_ranking(),
                'quality': self.get_quality_ranking(),
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Results exported to {filepath}")
        return filepath

    def export_csv(self, filename: Optional[str] = None) -> Path:
        """Export results to CSV"""
        import csv

        filename = filename or f"performance_{self.timestamp}.csv"
        filepath = self.output_dir / filename

        with open(filepath, 'w', newline='') as f:
            if not self.results:
                return filepath

            writer = csv.DictWriter(f, fieldnames=self.results[0].to_dict().keys())
            writer.writeheader()
            for result in self.results:
                writer.writerow(result.to_dict())

        print(f"Results exported to {filepath}")
        return filepath

    def plot_throughput_comparison(self, save_path: Optional[Path] = None):
        """Plot throughput comparison"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed")
            return

        algorithms = [r.algorithm for r in self.results]
        throughputs = [r.steps_per_second for r in self.results]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(algorithms, throughputs)
        ax.set_ylabel('Steps/Second')
        ax.set_title('Training Throughput Comparison')
        ax.grid(axis='y', alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_quality_comparison(self, save_path: Optional[Path] = None):
        """Plot reward quality comparison"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed")
            return

        algorithms = [r.algorithm for r in self.results]
        means = [r.mean_reward for r in self.results]
        stds = [r.std_reward for r in self.results]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(algorithms, means, yerr=stds, capsize=5)
        ax.set_ylabel('Mean Reward')
        ax.set_title('Reward Quality Comparison')
        ax.grid(axis='y', alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def generate_html_report(self, filename: Optional[str] = None) -> Path:
        """Generate HTML report"""
        filename = filename or f"performance_{self.timestamp}.html"
        filepath = self.output_dir / filename

        html = """
        <html>
        <head>
            <title>Performance Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #4CAF50; color: white; }
                .metric { margin: 20px 0; }
                h2 { color: #333; }
            </style>
        </head>
        <body>
            <h1>Performance Analysis Report</h1>
            <p>Generated: """ + self.timestamp + """</p>
        """

        # Results table
        html += "<h2>Results Summary</h2><table>"
        if self.results:
            html += "<tr><th>" + "</th><th>".join(self.results[0].to_dict().keys()) + "</th></tr>"
            for result in self.results:
                values = result.to_dict()
                html += "<tr><td>" + "</td><td>".join(str(v) for v in values.values()) + "</td></tr>"
        html += "</table>"

        # Rankings
        html += "<h2>Rankings</h2>"

        html += "<div class='metric'><h3>By Throughput</h3><ol>"
        for algo, sps in self.get_throughput_ranking():
            html += f"<li>{algo}: {sps:.0f} steps/sec</li>"
        html += "</ol></div>"

        html += "<div class='metric'><h3>By Reward Quality</h3><ol>"
        for algo, reward in self.get_quality_ranking():
            html += f"<li>{algo}: {reward:.2f}</li>"
        html += "</ol></div>"

        html += """
        </body>
        </html>
        """

        with open(filepath, 'w') as f:
            f.write(html)

        print(f"HTML report saved to {filepath}")
        return filepath


def create_performance_metrics_from_trainer(trainer) -> PerformanceMetrics:
    """
    Create PerformanceMetrics from a trainer object

    Args:
        trainer: RLTrainer instance after training

    Returns:
        PerformanceMetrics with computed metrics
    """
    from training.framework import RLTrainer

    if not isinstance(trainer, RLTrainer):
        raise TypeError("trainer must be an RLTrainer instance")

    metrics = trainer.get_metrics()
    total_timesteps = trainer.total_timesteps

    # Compute throughput
    analyzer = PerformanceAnalyzer()
    throughput = analyzer.compute_throughput(
        total_timesteps,
        trainer.metrics.get('training_time', 0)
    )

    # Fill in additional metrics
    throughput.algorithm = trainer.config.algorithm
    throughput.mean_reward = np.mean(metrics.get('episode_rewards', [0]))
    throughput.std_reward = np.std(metrics.get('episode_rewards', [0]))
    throughput.max_reward = np.max(metrics.get('episode_rewards', [0])) if metrics.get('episode_rewards') else 0

    if metrics.get('episode_antes'):
        throughput.mean_antes = np.mean(metrics['episode_antes'])
        throughput.max_antes = int(np.max(metrics['episode_antes']))
        throughput.success_rate = (
            sum(1 for a in metrics['episode_antes'] if a >= 5) / len(metrics['episode_antes']) * 100
        )

    return throughput
