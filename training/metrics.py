"""
Evaluation Metrics and Analysis
================================

Provides standardized metrics for evaluating RL agent performance
on Balatro.

Key Functions:
- evaluate_agent: Comprehensive agent evaluation
- compute_statistics: Statistical analysis of results
- plot_training_curves: Visualization of training progress
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from pathlib import Path


class BalatroMetrics:
    """Metrics specific to Balatro RL training"""

    def __init__(self):
        self.episode_rewards: List[float] = []
        self.episode_antes: List[int] = []
        self.episode_scores: List[float] = []
        self.episode_lengths: List[int] = []

    def add_episode(self, reward: float, ante: int, score: float, length: int):
        """Record an episode"""
        self.episode_rewards.append(reward)
        self.episode_antes.append(ante)
        self.episode_scores.append(score)
        self.episode_lengths.append(length)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.episode_rewards:
            return {}

        return {
            'mean_reward': float(np.mean(self.episode_rewards)),
            'std_reward': float(np.std(self.episode_rewards)),
            'max_reward': float(np.max(self.episode_rewards)),
            'min_reward': float(np.min(self.episode_rewards)),
            'mean_ante': float(np.mean(self.episode_antes)),
            'max_ante': int(np.max(self.episode_antes)),
            'mean_score': float(np.mean(self.episode_scores)),
            'max_score': float(np.max(self.episode_scores)),
            'mean_length': float(np.mean(self.episode_lengths)),
            'num_episodes': len(self.episode_rewards),
        }

    def get_percentiles(self) -> Dict[str, float]:
        """Get percentile statistics"""
        if not self.episode_rewards:
            return {}

        return {
            'reward_25th': float(np.percentile(self.episode_rewards, 25)),
            'reward_50th': float(np.percentile(self.episode_rewards, 50)),
            'reward_75th': float(np.percentile(self.episode_rewards, 75)),
            'reward_95th': float(np.percentile(self.episode_rewards, 95)),
            'ante_75th': float(np.percentile(self.episode_antes, 75)),
            'ante_90th': float(np.percentile(self.episode_antes, 90)),
            'ante_95th': float(np.percentile(self.episode_antes, 95)),
        }

    def success_rate(self, threshold: int = 5) -> float:
        """Percentage of episodes reaching ante threshold"""
        if not self.episode_antes:
            return 0.0
        return float(np.mean([a >= threshold for a in self.episode_antes]) * 100)

    def __str__(self) -> str:
        """String representation"""
        summary = self.get_summary()
        percentiles = self.get_percentiles()

        lines = [
            "Balatro Metrics Summary",
            "=" * 50,
            f"Episodes: {summary.get('num_episodes', 0)}",
            f"Mean Reward: {summary.get('mean_reward', 0):.2f} ± {summary.get('std_reward', 0):.2f}",
            f"Reward Range: [{summary.get('min_reward', 0):.2f}, {summary.get('max_reward', 0):.2f}]",
            f"Mean Ante: {summary.get('mean_ante', 0):.1f} (max: {summary.get('max_ante', 0)})",
            f"Mean Score: {summary.get('mean_score', 0):.0f} (max: {summary.get('max_score', 0):.0f})",
            f"Mean Episode Length: {summary.get('mean_length', 0):.0f}",
            f"Success Rate (ante ≥ 5): {self.success_rate(5):.1f}%",
        ]

        return "\n".join(lines)


def compute_moving_average(values: List[float], window: int = 100) -> np.ndarray:
    """Compute moving average"""
    if len(values) < window:
        window = len(values)

    return np.convolve(values, np.ones(window) / window, mode='valid')


def compute_success_window(antes: List[int], window: int = 100, threshold: int = 5) -> np.ndarray:
    """Compute success rate in sliding window"""
    if len(antes) < window:
        window = len(antes)

    success = [1 if a >= threshold else 0 for a in antes]
    return np.convolve(success, np.ones(window) / window, mode='valid') * 100


def print_comparison_table(results: Dict[str, Dict[str, float]]) -> str:
    """Print comparison table of algorithms"""

    lines = [
        "\n" + "=" * 100,
        "Algorithm Comparison",
        "=" * 100,
        f"{'Algorithm':<15} {'Mean Reward':<20} {'Mean Antes':<15} {'Success Rate':<15} {'Mean Score':<15}",
        "-" * 100,
    ]

    for algo, metrics in results.items():
        reward = f"{metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}"
        antes = f"{metrics['mean_ante']:.1f}"
        success = f"{metrics['success_rate']:.1f}%"
        score = f"{metrics['mean_score']:.0f}"

        lines.append(f"{algo:<15} {reward:<20} {antes:<15} {success:<15} {score:<15}")

    lines.append("=" * 100 + "\n")

    return "\n".join(lines)


def plot_training_curves(metrics_dict: Dict[str, Dict[str, List[float]]],
                        save_path: Optional[Path] = None) -> None:
    """
    Plot training curves for multiple algorithms

    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot generation.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Curves Comparison')

    # Plot 1: Reward over time
    ax = axes[0, 0]
    for algo, metrics in metrics_dict.items():
        if 'episode_rewards' in metrics:
            ax.plot(metrics['episode_rewards'], label=algo, alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Reward per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Moving average reward
    ax = axes[0, 1]
    for algo, metrics in metrics_dict.items():
        if 'episode_rewards' in metrics:
            ma = compute_moving_average(metrics['episode_rewards'], window=100)
            ax.plot(ma, label=algo, alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Moving Avg Reward (window=100)')
    ax.set_title('Smoothed Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Antes reached
    ax = axes[1, 0]
    for algo, metrics in metrics_dict.items():
        if 'episode_antes' in metrics:
            ax.plot(metrics['episode_antes'], label=algo, alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Antes Reached')
    ax.set_title('Antes per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Success rate
    ax = axes[1, 1]
    for algo, metrics in metrics_dict.items():
        if 'episode_antes' in metrics:
            success_rate = compute_success_window(metrics['episode_antes'], window=100, threshold=5)
            ax.plot(success_rate, label=algo, alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate (ante ≥ 5, window=100)')
    ax.set_ylim([0, 105])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def format_metrics_table(algorithm: str, metrics: Dict[str, Any]) -> str:
    """Format metrics as table"""
    lines = [
        f"\n{algorithm} - Metrics Summary",
        "=" * 60,
        f"Episodes: {metrics.get('num_episodes', 0)}",
        f"Mean Reward: {metrics.get('mean_reward', 0):.2f} ± {metrics.get('std_reward', 0):.2f}",
        f"Reward Range: [{metrics.get('min_reward', 0):.2f}, {metrics.get('max_reward', 0):.2f}]",
        f"Mean Antes: {metrics.get('mean_ante', 0):.1f} (max: {metrics.get('max_ante', 0)})",
        f"Mean Score: {metrics.get('mean_score', 0):.0f} (max: {metrics.get('max_score', 0):.0f})",
        f"Mean Episode Length: {metrics.get('mean_length', 0):.0f}",
        "=" * 60,
    ]
    return "\n".join(lines)
