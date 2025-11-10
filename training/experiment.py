"""
Experiment Management and Multi-Algorithm Benchmarking
======================================================

Provides tools for managing RL experiments, running multiple algorithms,
and comparing their performance.

Key Classes:
- ExperimentConfig: Configuration for a single experiment
- MultiAlgorithmBenchmark: Run multiple algorithms and compare results
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from training.framework import TrainerConfig, create_trainer, RLTrainer


# ---------------------------------------------------------------------------
# Experiment Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run"""

    # Experiment metadata
    name: str = "default_experiment"
    description: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().strftime('%Y%m%d_%H%M%S'))

    # Training configuration
    algorithm: str = "PPO"
    total_timesteps: int = 1_000_000
    seed: int = 42
    n_envs: int = 8

    # Training features
    use_curriculum: bool = False
    use_behavioral_cloning: bool = False

    # Logging
    use_wandb: bool = False
    wandb_project: str = "balatro-rl"
    wandb_tags: List[str] = field(default_factory=list)

    # Hyperparameter overrides
    hyperparams_override: Dict[str, Any] = field(default_factory=dict)

    # Paths
    save_dir: str = "models"

    def to_trainer_config(self) -> TrainerConfig:
        """Convert to TrainerConfig"""

        config = TrainerConfig(
            algorithm=self.algorithm,
            total_timesteps=self.total_timesteps,
            seed=self.seed,
            n_envs=self.n_envs,
            use_curriculum=self.use_curriculum,
            use_behavioral_cloning=self.use_behavioral_cloning,
            use_wandb=self.use_wandb,
            wandb_project=self.wandb_project,
            wandb_tags=self.wandb_tags,
            save_dir=self.save_dir
        )

        # Apply hyperparameter overrides
        if self.hyperparams_override:
            for key, value in self.hyperparams_override.items():
                if hasattr(config.hyperparams, key):
                    setattr(config.hyperparams, key, value)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary"""
        return cls(**data)

    def save(self, path: Path):
        """Save config to JSON"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        """Load config from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Experiment Results
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Results from a single experiment"""

    config: ExperimentConfig
    trainer: RLTrainer
    total_timesteps: int
    mean_reward: float
    std_reward: float
    mean_antes: float
    mean_scores: float
    training_duration: float  # seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'algorithm': self.config.algorithm,
            'name': self.config.name,
            'total_timesteps': self.total_timesteps,
            'mean_reward': float(self.mean_reward),
            'std_reward': float(self.std_reward),
            'mean_antes': float(self.mean_antes),
            'mean_scores': float(self.mean_scores),
            'training_duration': float(self.training_duration),
            'save_path': str(self.trainer.save_path),
        }

    def __str__(self) -> str:
        """String representation"""
        return f"""
Experiment: {self.config.algorithm}
  Name: {self.config.name}
  Total timesteps: {self.total_timesteps:,}
  Mean reward: {self.mean_reward:.2f} +/- {self.std_reward:.2f}
  Mean antes: {self.mean_antes:.1f}
  Mean scores: {self.mean_scores:.0f}
  Training time: {self.training_duration:.1f}s
  Save path: {self.trainer.save_path}
"""


# ---------------------------------------------------------------------------
# Multi-Algorithm Benchmark
# ---------------------------------------------------------------------------

class MultiAlgorithmBenchmark:
    """Run multiple algorithms and compare results"""

    def __init__(self, env_creator, benchmark_name: str = "balatro_benchmark"):
        """
        Initialize benchmark

        Args:
            env_creator: Callable that creates a new environment instance
            benchmark_name: Name for this benchmark run
        """
        self.env_creator = env_creator
        self.benchmark_name = benchmark_name
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.benchmark_dir = Path("benchmarks") / f"{benchmark_name}_{self.timestamp}"
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[ExperimentResult] = []
        self.configs: List[ExperimentConfig] = []

    def add_experiment(self, config: ExperimentConfig):
        """Add an experiment configuration"""
        self.configs.append(config)

    def add_experiments(self, configs: List[ExperimentConfig]):
        """Add multiple experiment configurations"""
        self.configs.extend(configs)

    def run_all(self, verbose: bool = True) -> List[ExperimentResult]:
        """
        Run all experiments

        Args:
            verbose: Print progress information

        Returns:
            List of ExperimentResult objects
        """
        import time

        self.results = []

        for i, config in enumerate(self.configs, 1):
            if verbose:
                print(f"\n{'='*70}")
                print(f"Experiment {i}/{len(self.configs)}: {config.algorithm}")
                print(f"{'='*70}")
                print(f"Name: {config.name}")
                print(f"Timesteps: {config.total_timesteps:,}")
                print(f"Seed: {config.seed}")

            # Create environment
            env = self.env_creator()

            # Convert to trainer config
            trainer_config = config.to_trainer_config()

            # Create and train
            try:
                start_time = time.time()

                trainer = create_trainer(env, trainer_config)
                trainer.train()

                # Evaluate
                mean_reward, std_reward = trainer.evaluate(n_episodes=config.total_timesteps // 100000)

                # Save results
                trainer.save_model()
                trainer.save_config()

                elapsed = time.time() - start_time

                # Extract metrics
                metrics = trainer.get_metrics()
                mean_antes = np.mean(metrics.get('episode_antes', [1]))
                mean_scores = np.mean(metrics.get('episode_scores', [0]))

                # Create result
                result = ExperimentResult(
                    config=config,
                    trainer=trainer,
                    total_timesteps=trainer.total_timesteps,
                    mean_reward=mean_reward,
                    std_reward=std_reward,
                    mean_antes=mean_antes,
                    mean_scores=mean_scores,
                    training_duration=elapsed
                )

                self.results.append(result)

                if verbose:
                    print(f"✓ Completed in {elapsed:.1f}s")
                    print(f"  Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
                    print(f"  Mean antes: {mean_antes:.1f}")
                    print(f"  Mean scores: {mean_scores:.0f}")

                trainer.close()

            except Exception as e:
                print(f"✗ Failed with error: {e}")
                import traceback
                traceback.print_exc()

        # Save benchmark results
        self._save_results()

        if verbose:
            print(f"\n{'='*70}")
            print(f"Benchmark Complete: {self.benchmark_name}")
            print(f"{'='*70}")
            self.print_summary()

        return self.results

    def _save_results(self):
        """Save benchmark results to JSON"""
        results_data = {
            'benchmark_name': self.benchmark_name,
            'timestamp': self.timestamp,
            'num_experiments': len(self.results),
            'results': [r.to_dict() for r in self.results]
        }

        save_path = self.benchmark_dir / "results.json"
        with open(save_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        # Also save as human-readable text
        text_path = self.benchmark_dir / "results.txt"
        with open(text_path, 'w') as f:
            f.write(f"Benchmark: {self.benchmark_name}\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Number of experiments: {len(self.results)}\n")
            f.write(f"{'='*70}\n\n")

            for result in self.results:
                f.write(str(result))
                f.write("\n")

        if hasattr(self, 'benchmark_dir'):
            print(f"Results saved to: {self.benchmark_dir}")

    def print_summary(self):
        """Print summary of all results"""
        if not self.results:
            print("No results to summarize")
            return

        print("\n" + "="*70)
        print("Algorithm Comparison Summary")
        print("="*70)
        print(f"{'Algorithm':<12} {'Mean Reward':<15} {'Mean Antes':<15} {'Mean Score':<15}")
        print("-"*70)

        for result in sorted(self.results, key=lambda r: r.mean_reward, reverse=True):
            algo = result.config.algorithm[:12]
            reward = f"{result.mean_reward:.2f} ± {result.std_reward:.2f}"
            antes = f"{result.mean_antes:.1f}"
            scores = f"{result.mean_scores:.0f}"
            print(f"{algo:<12} {reward:<15} {antes:<15} {scores:<15}")

        print("="*70)

    def get_best_result(self, metric: str = 'mean_reward') -> Optional[ExperimentResult]:
        """Get the best result by metric"""
        if not self.results:
            return None

        if metric == 'mean_reward':
            return max(self.results, key=lambda r: r.mean_reward)
        elif metric == 'mean_antes':
            return max(self.results, key=lambda r: r.mean_antes)
        elif metric == 'mean_scores':
            return max(self.results, key=lambda r: r.mean_scores)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def compare_algorithms(self) -> Dict[str, Any]:
        """Get comparison data"""
        return {
            'algorithms': [r.config.algorithm for r in self.results],
            'mean_rewards': [r.mean_reward for r in self.results],
            'std_rewards': [r.std_reward for r in self.results],
            'mean_antes': [r.mean_antes for r in self.results],
            'mean_scores': [r.mean_scores for r in self.results],
            'training_times': [r.training_duration for r in self.results],
        }


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def create_standard_comparison() -> List[ExperimentConfig]:
    """Create standard comparison experiments"""

    experiments = [
        ExperimentConfig(
            name="PPO_baseline",
            algorithm="PPO",
            total_timesteps=100_000,
            seed=42,
            n_envs=4
        ),
        ExperimentConfig(
            name="A2C_baseline",
            algorithm="A2C",
            total_timesteps=100_000,
            seed=42,
            n_envs=4
        ),
        ExperimentConfig(
            name="DQN_baseline",
            algorithm="DQN",
            total_timesteps=100_000,
            seed=42,
            n_envs=1
        ),
    ]

    return experiments
