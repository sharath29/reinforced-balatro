#!/usr/bin/env python3
"""
Training Script Using Framework
================================

Complete training script that leverages the new training framework
for maximum flexibility and clean code organization.

Features:
- Multi-algorithm support (PPO, A2C, DQN)
- Curriculum learning
- Behavioral cloning warm-start
- Comprehensive logging
- Model checkpointing
- Hyperparameter tuning

Usage:
    # Basic training with PPO
    python train_with_framework.py

    # Train with A2C
    python train_with_framework.py --algorithm A2C --timesteps 500000

    # With curriculum learning
    python train_with_framework.py --curriculum --timesteps 1000000

    # Quick test (10k steps)
    python train_with_framework.py --quick-test

    # Hyperparameter override
    python train_with_framework.py --lr 5e-4 --batch-size 128
"""

import argparse
from pathlib import Path
import sys
import json

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from training.framework import TrainerConfig, create_trainer, get_default_config, HyperparameterConfig
from training.utils import (
    create_balatro_env,
    make_vectorized_env,
    validate_environment,
    print_config
)
from training.metrics import BalatroMetrics, format_metrics_table
from training.env_wrapper import wrap_balatro_env


def main():
    parser = argparse.ArgumentParser(
        description="Train RL agents on Balatro using the framework"
    )

    # Algorithm and training parameters
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
        default=1_000_000,
        help="Total training timesteps (default: 1,000,000)"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments (default: 8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    # Feature and preprocessing options
    parser.add_argument(
        "--use-curriculum",
        action="store_true",
        help="Enable curriculum learning"
    )
    parser.add_argument(
        "--normalize-obs",
        action="store_true",
        default=True,
        help="Normalize observations (default: True)"
    )
    parser.add_argument(
        "--normalize-reward",
        action="store_true",
        default=True,
        help="Normalize rewards (default: True)"
    )

    # Logging and monitoring
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="balatro-rl",
        help="W&B project name (default: balatro-rl)"
    )
    parser.add_argument(
        "--wandb-tags",
        nargs="+",
        default=[],
        help="W&B tags"
    )

    # Checkpointing
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models",
        help="Directory to save models (default: models)"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10_000,
        help="Save checkpoint every N timesteps (default: 10,000)"
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5_000,
        help="Evaluate every N timesteps (default: 5,000)"
    )

    # Hyperparameter overrides
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides default)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Discount factor (overrides default)"
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=None,
        help="Entropy coefficient (overrides default)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides default)"
    )

    # Special modes
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test mode (10k timesteps, 2 envs)"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load model and evaluate only"
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Path to model to load for evaluation"
    )

    args = parser.parse_args()

    # Quick test mode overrides
    if args.quick_test:
        args.timesteps = 10_000
        args.n_envs = 2
        args.checkpoint_freq = 5_000
        args.eval_freq = 5_000

    # Print configuration
    print(f"\n{'='*70}")
    print(f"Balatro RL Training - {args.algorithm}")
    print(f"{'='*70}\n")

    # Create base config
    config = get_default_config(args.algorithm)

    # Apply command-line overrides
    config.total_timesteps = args.timesteps
    config.n_envs = args.n_envs
    config.seed = args.seed
    config.save_dir = args.save_dir
    config.checkpoint_freq = args.checkpoint_freq
    config.eval_freq = args.eval_freq
    config.use_curriculum = args.use_curriculum
    config.normalize_obs = args.normalize_obs
    config.normalize_reward = args.normalize_reward
    config.use_wandb = not args.no_wandb
    config.wandb_project = args.wandb_project
    config.wandb_tags = args.wandb_tags

    # Apply hyperparameter overrides
    if args.lr:
        config.hyperparams.learning_rate = args.lr
    if args.gamma:
        config.hyperparams.gamma = args.gamma
    if args.entropy_coef:
        config.hyperparams.entropy_coefficient = args.entropy_coef
    if args.batch_size:
        config.hyperparams.algorithm_specific['batch_size'] = args.batch_size

    # Print config
    print_config(config)

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

    # Validate environment
    if not validate_environment(env):
        sys.exit(1)

    # Wrap environment
    if args.use_curriculum:
        print("Applying curriculum learning wrapper...")
        env = wrap_balatro_env(env, use_curriculum=True)

    # Create trainer
    print("Creating trainer...")
    trainer = create_trainer(env, config)

    # Training
    if not args.eval_only:
        print("Starting training...\n")
        try:
            result = trainer.train()

            print(f"\n{'='*70}")
            print(f"Training Complete")
            print(f"{'='*70}")
            print(f"Total timesteps: {result['total_timesteps']:,}")
            print(f"Mean reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
            print(f"Model saved to: {trainer.save_path}")

        except KeyboardInterrupt:
            print(f"\n✗ Training interrupted by user")
            print(f"Model saved to: {trainer.save_path}")

    # Evaluation
    if not args.eval_only or args.load_model:
        print("\nStarting evaluation...")
        n_eval_episodes = max(10, args.timesteps // 50000)
        mean_reward, std_reward = trainer.evaluate(n_episodes=n_eval_episodes, deterministic=True)

        print(f"\n{'='*70}")
        print(f"Evaluation Results")
        print(f"{'='*70}")
        print(f"Episodes: {n_eval_episodes}")
        print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    # Print detailed metrics
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

        print("\n" + str(metrics_obj))
        print(f"\nDetailed Metrics:")
        print(format_metrics_table(args.algorithm, metrics_obj.get_summary()))

    # Save final model
    if not args.eval_only:
        model_path = trainer.save_model()
        print(f"\nFinal model saved to: {model_path}")

    # Save configuration
    if not args.eval_only:
        trainer.save_config()
        config_path = trainer.save_path / "config.json"
        print(f"Configuration saved to: {config_path}")

    # Summary
    trainer.print_summary()

    # Cleanup
    trainer.close()

    print(f"\n✓ Complete! Results saved to: {trainer.save_path}")


if __name__ == "__main__":
    main()
