#!/usr/bin/env python3
"""
Algorithm Comparison Script
===========================

Compare multiple RL algorithms on the same environment with fair conditions.
Runs experiments sequentially and produces comparison visualizations.

Usage:
    # Compare all three algorithms
    python compare_algorithms.py

    # Compare PPO and A2C only
    python compare_algorithms.py --algorithms PPO A2C

    # Custom timesteps
    python compare_algorithms.py --timesteps 500000

    # No W&B logging
    python compare_algorithms.py --no-wandb

    # Shorter evaluation for quick testing
    python compare_algorithms.py --quick
"""

import argparse
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from training.experiment import ExperimentConfig, MultiAlgorithmBenchmark
from training.utils import create_balatro_env
from training.metrics import print_comparison_table


def main():
    parser = argparse.ArgumentParser(
        description="Compare RL algorithms on Balatro"
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["PPO", "A2C", "DQN"],
        choices=["PPO", "A2C", "DQN"],
        help="Algorithms to compare (default: all three)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps per algorithm (default: 100,000)"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (10k timesteps, minimal evaluation)"
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable curriculum learning for all experiments"
    )
    parser.add_argument(
        "--benchmark-name",
        type=str,
        default="balatro_comparison",
        help="Name for this benchmark run"
    )

    args = parser.parse_args()

    # Override for quick mode
    if args.quick:
        args.timesteps = 10_000
        print("Quick mode: Using 10k timesteps")

    print(f"\n{'='*70}")
    print(f"Algorithm Comparison Benchmark")
    print(f"{'='*70}")
    print(f"Algorithms: {', '.join(args.algorithms)}")
    print(f"Timesteps per algorithm: {args.timesteps:,}")
    print(f"Parallel environments: {args.n_envs}")
    print(f"Seed: {args.seed}")
    print(f"Curriculum learning: {args.curriculum}")
    print(f"W&B Logging: {not args.no_wandb}")
    print(f"{'='*70}\n")

    # Create experiment configs
    configs = []
    for algo in args.algorithms:
        config = ExperimentConfig(
            name=f"{algo}_comparison",
            algorithm=algo,
            total_timesteps=args.timesteps,
            seed=args.seed,
            n_envs=args.n_envs,
            use_wandb=not args.no_wandb,
            use_curriculum=args.curriculum,
            wandb_tags=["comparison", args.benchmark_name]
        )
        configs.append(config)

    # Create benchmark
    benchmark = MultiAlgorithmBenchmark(
        env_creator=create_balatro_env,
        benchmark_name=args.benchmark_name
    )
    benchmark.add_experiments(configs)

    # Run all experiments
    print("Starting benchmark...\n")
    results = benchmark.run_all(verbose=True)

    # Print comparison
    if results:
        comparison_data = {}
        for result in results:
            comparison_data[result.config.algorithm] = {
                'mean_reward': result.mean_reward,
                'std_reward': result.std_reward,
                'mean_ante': result.mean_antes,
                'mean_score': result.mean_scores,
                'success_rate': (
                    sum(1 for a in result.trainer.get_metrics().get('episode_antes', [1]) if a >= 5)
                    / max(1, len(result.trainer.get_metrics().get('episode_antes', [1])))
                    * 100
                )
            }

        print(print_comparison_table(comparison_data))

        # Get best algorithm
        best_result = benchmark.get_best_result(metric='mean_reward')
        if best_result:
            print(f"Best algorithm by reward: {best_result.config.algorithm}")
            print(f"  Mean reward: {best_result.mean_reward:.2f} ± {best_result.std_reward:.2f}")
            print(f"  Model path: {best_result.trainer.save_path}")

    print(f"\nBenchmark results saved to: {benchmark.benchmark_dir}")

    # Try to plot if matplotlib available
    try:
        import matplotlib.pyplot as plt
        from training.metrics import plot_training_curves

        metrics_dict = {}
        for result in results:
            metrics = result.trainer.get_metrics()
            metrics_dict[result.config.algorithm] = metrics

        plot_save_path = benchmark.benchmark_dir / "comparison.png"
        plot_training_curves(metrics_dict, save_path=plot_save_path)
    except ImportError:
        print("\nNote: Install matplotlib for training curve visualization")
    except Exception as e:
        print(f"\nNote: Could not generate plots: {e}")


if __name__ == "__main__":
    main()
