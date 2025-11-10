"""
Training Framework for Balatro RL
==================================

This module provides a flexible, multi-algorithm training framework
that abstracts algorithm differences and enables easy experimentation
with different RL algorithms (PPO, A2C, DQN, etc.).

Key Components:
- framework.py: Base RLTrainer class and TrainerConfig
- sb3_trainer.py: Stable-Baselines3 implementation
- experiment.py: Experiment configuration and multi-algorithm benchmarking
- metrics.py: Unified evaluation metrics
- utils.py: Helper functions for logging and visualization
"""

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "RLTrainer":
        from training.framework import RLTrainer
        return RLTrainer
    elif name == "TrainerConfig":
        from training.framework import TrainerConfig
        return TrainerConfig
    elif name == "SB3Trainer":
        from training.sb3_trainer import SB3Trainer
        return SB3Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "RLTrainer",
    "TrainerConfig",
    "SB3Trainer",
]
