"""
Core Training Framework
=======================

Provides abstract base classes for RL trainers and unified configuration.
This enables algorithm-agnostic training and easy switching between
different RL algorithms.

Key Classes:
- TrainerConfig: Unified hyperparameter and configuration management
- RLTrainer: Abstract base class for all training implementations
- CheckpointManager: Handles model saving/loading and checkpointing
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json
from datetime import datetime
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Configuration Classes
# ---------------------------------------------------------------------------

@dataclass
class HyperparameterConfig:
    """Hyperparameter configuration for RL algorithms"""

    # Common hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    entropy_coefficient: float = 0.01
    value_function_coefficient: float = 0.5

    # Algorithm-specific parameters (as dict)
    algorithm_specific: Dict[str, Any] = field(default_factory=dict)

    # Policy network architecture
    policy_net_arch: List[int] = field(default_factory=lambda: [256, 256])
    value_net_arch: List[int] = field(default_factory=lambda: [256, 256])
    features_dim: int = 512

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HyperparameterConfig":
        """Create from dictionary"""
        return cls(**data)


@dataclass
class TrainerConfig:
    """Unified training configuration for all algorithms"""

    # Algorithm selection
    algorithm: str = "PPO"  # Options: PPO, A2C, DQN

    # Training parameters
    total_timesteps: int = 1_000_000
    seed: int = 42
    n_envs: int = 8

    # Saving and logging
    save_dir: str = "models"
    checkpoint_freq: int = 10_000
    eval_freq: int = 5_000
    eval_episodes: int = 10

    # Features and preprocessing
    use_feature_extractor: bool = True
    feature_extractor_type: str = "balatro"  # Custom feature extractor for Balatro
    normalize_obs: bool = True
    normalize_reward: bool = True

    # Training features
    use_curriculum: bool = False
    use_behavioral_cloning: bool = False
    behavioral_cloning_epochs: int = 10

    # Logging and monitoring
    use_wandb: bool = False
    wandb_project: str = "balatro-rl"
    wandb_tags: List[str] = field(default_factory=list)
    log_interval: int = 10

    # Hyperparameters
    hyperparams: HyperparameterConfig = field(default_factory=HyperparameterConfig)

    # Reproducibility
    deterministic: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        config_dict = asdict(self)
        config_dict['hyperparams'] = self.hyperparams.to_dict()
        return config_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainerConfig":
        """Create from dictionary"""
        hyperparams_data = data.pop('hyperparams', {})
        hyperparams = HyperparameterConfig.from_dict(hyperparams_data)
        return cls(hyperparams=hyperparams, **data)

    def save(self, path: Path):
        """Save config to JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TrainerConfig":
        """Load config from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Checkpoint Manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Handles model checkpointing and recovery"""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_score = None
        self.best_checkpoint = None

    def save_checkpoint(self, model: Any, timestep: int, score: Optional[float] = None):
        """Save a checkpoint with optional scoring"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{timestep:09d}"
        checkpoint_path.mkdir(exist_ok=True)

        # Save model (implementation-specific)
        model_path = checkpoint_path / "model"
        if hasattr(model, 'save'):
            model.save(str(model_path))

        # Track best checkpoint if score provided
        if score is not None:
            if self.best_score is None or score > self.best_score:
                self.best_score = score
                self.best_checkpoint = checkpoint_path
                # Create symlink to best
                best_link = self.checkpoint_dir / "best_model"
                if best_link.exists():
                    best_link.unlink()
                best_link.symlink_to(checkpoint_path)

        return checkpoint_path

    def load_checkpoint(self, timestep: int) -> Path:
        """Load a checkpoint by timestep"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{timestep:09d}"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the most recent checkpoint"""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*"))
        return checkpoints[-1] if checkpoints else None

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get the best checkpoint by score"""
        return self.best_checkpoint


# ---------------------------------------------------------------------------
# Base Trainer Class
# ---------------------------------------------------------------------------

class RLTrainer(ABC):
    """
    Abstract base class for RL trainers.

    Defines the interface that all algorithm implementations must follow,
    enabling seamless switching between different RL algorithms.
    """

    def __init__(self,
                 env: gym.Env,
                 config: TrainerConfig):
        """
        Initialize trainer

        Args:
            env: Gymnasium environment
            config: TrainerConfig instance
        """
        self.env = env
        self.config = config
        self.model = None
        self.total_timesteps = 0

        # Setup paths
        self.save_path = Path(config.save_dir) / f"{config.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_path.mkdir(parents=True, exist_ok=True)

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(self.save_path / "checkpoints")

        # Initialize metrics
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_antes': [],
            'episode_scores': [],
            'training_steps': [],
            'mean_reward': [],
            'std_reward': []
        }

    @abstractmethod
    def create_model(self) -> Any:
        """
        Create the RL model

        Must be implemented by subclasses
        """
        pass

    @abstractmethod
    def train(self,
              total_timesteps: Optional[int] = None,
              callback: Optional[Any] = None) -> Dict[str, Any]:
        """
        Train the model

        Args:
            total_timesteps: Number of timesteps to train (uses config if None)
            callback: Optional callback for training

        Returns:
            Dictionary with training statistics
        """
        pass

    @abstractmethod
    def evaluate(self,
                 n_episodes: int,
                 deterministic: bool = True) -> Tuple[float, float]:
        """
        Evaluate the current model

        Args:
            n_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic policy

        Returns:
            Tuple of (mean_reward, std_reward)
        """
        pass

    def save_model(self, name: str = "final") -> Path:
        """Save the trained model"""
        if self.model is None:
            raise RuntimeError("No model to save. Train first.")

        model_path = self.save_path / f"{self.config.algorithm}_{name}"
        if hasattr(self.model, 'save'):
            self.model.save(str(model_path))

        return model_path

    def save_checkpoint(self, score: Optional[float] = None) -> Path:
        """Save a checkpoint"""
        if self.model is None:
            raise RuntimeError("No model to checkpoint. Train first.")

        return self.checkpoint_manager.save_checkpoint(
            self.model,
            self.total_timesteps,
            score
        )

    def save_config(self):
        """Save training configuration"""
        self.config.save(self.save_path / "config.json")

    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics"""
        return self.metrics.copy()

    def get_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        return {
            'algorithm': self.config.algorithm,
            'total_timesteps': self.total_timesteps,
            'save_path': str(self.save_path),
            'config': self.config.to_dict(),
            'metrics': self.metrics
        }

    def print_summary(self):
        """Print training summary"""
        print(f"\n{'='*70}")
        print(f"Training Summary: {self.config.algorithm}")
        print(f"{'='*70}")
        print(f"Total timesteps: {self.total_timesteps:,}")
        print(f"Save path: {self.save_path}")
        print(f"Config saved: {self.save_path / 'config.json'}")

        if self.metrics['episode_rewards']:
            mean_reward = np.mean(self.metrics['episode_rewards'])
            std_reward = np.std(self.metrics['episode_rewards'])
            print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        if self.metrics['episode_antes']:
            mean_ante = np.mean(self.metrics['episode_antes'])
            print(f"Mean antes reached: {mean_ante:.1f}")

        if self.metrics['episode_scores']:
            mean_score = np.mean(self.metrics['episode_scores'])
            print(f"Mean score: {mean_score:.0f}")

        print(f"{'='*70}\n")

    def close(self):
        """Close environment and cleanup"""
        if hasattr(self.env, 'close'):
            self.env.close()


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def create_trainer(env: gym.Env, config: TrainerConfig) -> RLTrainer:
    """
    Factory function to create appropriate trainer based on config

    Args:
        env: Gymnasium environment
        config: TrainerConfig instance

    Returns:
        RLTrainer instance

    Raises:
        ValueError: If algorithm is not supported
    """
    from training.sb3_trainer import SB3Trainer

    if config.algorithm in ["PPO", "A2C", "DQN"]:
        return SB3Trainer(env, config)
    else:
        raise ValueError(f"Unsupported algorithm: {config.algorithm}")


def get_default_config(algorithm: str = "PPO") -> TrainerConfig:
    """Get default training configuration for an algorithm"""

    default_hyperparams = {
        "PPO": {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "algorithm_specific": {
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "clip_range": 0.2,
            }
        },
        "A2C": {
            "learning_rate": 7e-4,
            "gamma": 0.99,
            "gae_lambda": 1.0,
            "algorithm_specific": {
                "n_steps": 5,
            }
        },
        "DQN": {
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "algorithm_specific": {
                "buffer_size": 100_000,
                "learning_starts": 10_000,
                "batch_size": 32,
                "train_freq": 4,
            }
        }
    }

    hp_config = default_hyperparams.get(algorithm, {})
    hyperparams = HyperparameterConfig(**hp_config)

    config = TrainerConfig(
        algorithm=algorithm,
        hyperparams=hyperparams
    )

    return config
