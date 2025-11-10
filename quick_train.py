#!/usr/bin/env python3
"""
Quick Training Script
====================

Fast prototyping script for single-algorithm training with sensible defaults.
Perfect for quick experiments and testing.

Usage:
    # Train with PPO (default)
    python quick_train.py

    # Train with A2C
    python quick_train.py --algorithm A2C

    # Train with custom timesteps
    python quick_train.py --timesteps 500000

    # No W&B logging
    python quick_train.py --no-wandb
"""

import argparse
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from training.framework import TrainerConfig, create_trainer, get_default_config
from training.utils import create_balatro_env, make_vectorized_env
from training.metrics import BalatroMetrics, format_metrics_table


def main():
    parser = argparse.ArgumentParser(
        description="Quick training with RL algorithms on Balatro"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        choices=["PPO", "A2C", "DQN"],
        help="RL algorithm to use (default: PPO)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps (default: 100,000)"
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
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models",
        help="Directory to save models (default: models)"
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable curriculum learning"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate, don't train"
    )

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Quick Training: {args.algorithm}")
    print(f"{'='*70}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Environments: {args.n_envs}")
    print(f"Seed: {args.seed}")
    print(f"Curriculum: {args.curriculum}")
    print(f"W&B Logging: {not args.no_wandb}")
    print(f"{'='*70}\n")

    # Get default config and customize
    config = get_default_config(args.algorithm)
    config.total_timesteps = args.timesteps
    config.n_envs = args.n_envs
    config.seed = args.seed
    config.use_wandb = not args.no_wandb
    config.use_curriculum = args.curriculum
    config.save_dir = args.save_dir

    # Create environment
    print("Creating environment...")
    if args.n_envs > 1:
        env = make_vectorized_env(
            create_balatro_env,
            n_envs=args.n_envs,
            seed=args.seed,
            use_subprocess=True
        )
    else:
        env = create_balatro_env(seed=args.seed)

    # Create trainer
    print("Creating trainer...")
    trainer = create_trainer(env, config)

    if not args.eval_only:
        # Train
        print("Starting training...")
        try:
            result = trainer.train()
            print("\n✓ Training complete!")
            print(f"  Total timesteps: {result['total_timesteps']:,}")
            print(f"  Mean reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
        except KeyboardInterrupt:
            print("\n✗ Training interrupted by user")

    # Evaluate
    print("\nEvaluating agent...")
    n_eval_episodes = min(20, max(1, args.timesteps // 10000))
    mean_reward, std_reward = trainer.evaluate(n_episodes=n_eval_episodes)
    print(f"✓ Evaluation complete!")
    print(f"  Episodes: {n_eval_episodes}")
    print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    # Save model
    model_path = trainer.save_model()
    print(f"\n✓ Model saved to: {model_path}")

    # Print summary
    trainer.print_summary()

    # Print metrics if available
    metrics = trainer.get_metrics()
    if metrics['episode_antes']:
        metrics_obj = BalatroMetrics()
        for reward, ante, score, length in zip(
            metrics.get('episode_rewards', []),
            metrics.get('episode_antes', []),
            metrics.get('episode_scores', []),
            metrics.get('episode_lengths', [])
        ):
            metrics_obj.add_episode(reward, ante, score, length)

        print(format_metrics_table(args.algorithm, metrics_obj.get_summary()))

    # Close
    trainer.close()


if __name__ == "__main__":
    main()
