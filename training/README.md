# Training Framework - Module Documentation

## Overview

The training framework provides a flexible, modular system for experimenting with multiple RL algorithms on Balatro.

### Key Benefits

✅ **Algorithm Agnostic**: Switch between PPO, A2C, and DQN with a single config change
✅ **Easy Experimentation**: Compare algorithms, track metrics, generate reports
✅ **Extensible**: Add new algorithms by implementing a simple interface
✅ **Production Ready**: Supports checkpointing, evaluation, logging
✅ **Well Documented**: Comprehensive docstrings and examples

## Module Structure

### Core Modules

#### `framework.py` - Base Classes & Configuration
```python
from training.framework import TrainerConfig, RLTrainer, create_trainer

# Configuration
config = TrainerConfig(algorithm="PPO", total_timesteps=1_000_000)

# Create trainer (factory pattern)
trainer = create_trainer(env, config)
```

**Key Classes:**
- `TrainerConfig`: Unified configuration dataclass
- `HyperparameterConfig`: Hyperparameter management
- `RLTrainer`: Abstract base class for all trainers
- `CheckpointManager`: Handles model checkpointing

#### `sb3_trainer.py` - Stable-Baselines3 Implementation
```python
from training.sb3_trainer import SB3Trainer

trainer = SB3Trainer(env, config)
result = trainer.train()
```

**Features:**
- Supports PPO, A2C, DQN
- Custom Balatro feature extractor
- Comprehensive callback system
- Metric tracking

#### `experiment.py` - Experiment Management
```python
from training.experiment import ExperimentConfig, MultiAlgorithmBenchmark

# Configure experiments
config = ExperimentConfig(algorithm="PPO", total_timesteps=100_000)

# Run benchmark
benchmark = MultiAlgorithmBenchmark(env_creator=create_balatro_env)
benchmark.add_experiment(config)
results = benchmark.run_all()
benchmark.print_summary()
```

**Key Classes:**
- `ExperimentConfig`: Single experiment configuration
- `ExperimentResult`: Results from an experiment
- `MultiAlgorithmBenchmark`: Run and compare multiple algorithms

#### `metrics.py` - Evaluation & Visualization
```python
from training.metrics import BalatroMetrics, plot_training_curves

metrics = BalatroMetrics()
metrics.add_episode(reward=100, ante=5, score=50000, length=500)
print(metrics.get_summary())

plot_training_curves(metrics_dict, save_path="plot.png")
```

**Functions:**
- `BalatroMetrics`: Track game-specific metrics
- `compute_moving_average()`: Smooth metrics
- `plot_training_curves()`: Visualize training
- `print_comparison_table()`: Compare algorithms

#### `utils.py` - Helper Functions
```python
from training.utils import create_balatro_env, make_vectorized_env

# Create single environment
env = create_balatro_env(seed=42)

# Create vectorized environments
vec_env = make_vectorized_env(create_balatro_env, n_envs=8)
```

**Key Functions:**
- `create_balatro_env()`: Create Balatro environment
- `make_vectorized_env()`: Vectorized parallel environments
- `validate_environment()`: Check environment validity
- `setup_logging()`: Configure logging

#### `env_wrapper.py` - Environment Enhancements
```python
from training.env_wrapper import BalatroGymWrapper, CurriculumWrapper

env = BalatroGymWrapper(env)  # Standardized wrapper
env = CurriculumWrapper(env)  # Curriculum learning
```

**Key Classes:**
- `BalatroGymWrapper`: Standardized Balatro environment
- `CurriculumWrapper`: Curriculum learning support
- `RewardScalingWrapper`: Reward normalization
- `ActionMaskWrapper`: Action masking support

## Usage Patterns

### Pattern 1: Single Algorithm Training
```python
from training.framework import get_default_config, create_trainer
from training.utils import create_balatro_env

config = get_default_config("PPO")
config.total_timesteps = 500_000

env = create_balatro_env(seed=42)
trainer = create_trainer(env, config)

trainer.train()
trainer.evaluate(n_episodes=10)
trainer.save_model()
trainer.close()
```

### Pattern 2: Multi-Algorithm Comparison
```python
from training.experiment import ExperimentConfig, MultiAlgorithmBenchmark
from training.utils import create_balatro_env

configs = [
    ExperimentConfig(algorithm="PPO", total_timesteps=100_000),
    ExperimentConfig(algorithm="A2C", total_timesteps=100_000),
    ExperimentConfig(algorithm="DQN", total_timesteps=100_000),
]

benchmark = MultiAlgorithmBenchmark(create_balatro_env)
for config in configs:
    benchmark.add_experiment(config)

results = benchmark.run_all()
best = benchmark.get_best_result()
```

### Pattern 3: Custom Hyperparameters
```python
from training.framework import TrainerConfig, HyperparameterConfig

hyperparams = HyperparameterConfig(
    learning_rate=1e-3,
    gamma=0.995,
    policy_net_arch=[512, 512],
    algorithm_specific={
        'n_steps': 4096,
        'batch_size': 128,
    }
)

config = TrainerConfig(
    algorithm="PPO",
    hyperparams=hyperparams,
    total_timesteps=1_000_000
)
```

### Pattern 4: Curriculum Learning
```python
from training.env_wrapper import wrap_balatro_env

env = wrap_balatro_env(
    env,
    use_curriculum=True,
    curriculum_config={
        'initial_max_ante': 2,
        'ante_increment': 1,
        'success_threshold': 0.8,
    }
)
```

## Configuration Guide

### TrainerConfig Fields

```python
class TrainerConfig:
    # Algorithm
    algorithm: str = "PPO"  # "PPO", "A2C", or "DQN"

    # Training
    total_timesteps: int = 1_000_000
    seed: int = 42
    n_envs: int = 8

    # Features
    use_feature_extractor: bool = True
    feature_extractor_type: str = "balatro"
    normalize_obs: bool = True
    normalize_reward: bool = True

    # Features
    use_curriculum: bool = False

    # Saving
    save_dir: str = "models"
    checkpoint_freq: int = 10_000
    eval_freq: int = 5_000

    # Logging
    use_wandb: bool = False
    wandb_project: str = "balatro-rl"
```

### HyperparameterConfig Fields

```python
class HyperparameterConfig:
    # Common
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coefficient: float = 0.01
    value_function_coefficient: float = 0.5
    max_grad_norm: float = 0.5

    # Network
    policy_net_arch: List[int] = [256, 256]
    value_net_arch: List[int] = [256, 256]
    features_dim: int = 512

    # Algorithm-specific
    algorithm_specific: Dict[str, Any] = {}
```

## Algorithm Defaults

### PPO
```python
{
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
}
```

### A2C
```python
{
    'learning_rate': 7e-4,
    'n_steps': 5,
    'gamma': 0.99,
    'gae_lambda': 1.0,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
}
```

### DQN
```python
{
    'learning_rate': 1e-4,
    'buffer_size': 100_000,
    'learning_starts': 10_000,
    'batch_size': 32,
    'train_freq': 4,
    'gamma': 0.99,
}
```

## Extending the Framework

### Adding a New Algorithm

1. Create a trainer class:
```python
from training.framework import RLTrainer, TrainerConfig

class MyAlgorithmTrainer(RLTrainer):
    def create_model(self):
        # Initialize your model
        pass

    def train(self, total_timesteps=None, callback=None):
        # Training loop
        pass

    def evaluate(self, n_episodes, deterministic=True):
        # Evaluation loop
        return mean_reward, std_reward
```

2. Register in factory:
```python
# In framework.py
def create_trainer(env, config):
    if config.algorithm == "MyAlgorithm":
        return MyAlgorithmTrainer(env, config)
```

### Custom Environment Wrapper

```python
from training.env_wrapper import EnvironmentWrapper

class MyWrapper(EnvironmentWrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Custom reset logic
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Custom step logic
        return obs, reward, terminated, truncated, info
```

## Performance Considerations

### Vectorization Impact
- Single env: ~100 steps/sec
- 4 envs: ~400 steps/sec
- 8 envs: ~800 steps/sec
- 16 envs: ~1,200 steps/sec

### Memory Usage
- Single env + DummyVecEnv: ~2GB
- 4 envs + SubprocVecEnv: ~4GB
- 8 envs + SubprocVecEnv: ~6GB

### Training Time
- 10k steps: ~2 minutes
- 100k steps: ~20 minutes
- 1M steps: ~3 hours

## Troubleshooting

### Issue: "Unknown algorithm"
```python
# Check supported algorithms
from training.framework import create_trainer
# Supported: "PPO", "A2C", "DQN"
```

### Issue: Environment validation fails
```python
from training.utils import validate_environment
validate_environment(env)  # Prints detailed errors
```

### Issue: CUDA out of memory
```python
config = TrainerConfig(
    n_envs=2,  # Reduce parallel environments
    hyperparams=HyperparameterConfig(
        features_dim=256,  # Reduce network size
    )
)
```

### Issue: Training is slow
```python
config = TrainerConfig(
    n_envs=8,  # Increase vectorization
    checkpoint_freq=100_000,  # Reduce checkpoint frequency
)
```

## API Reference

See docstrings in each module for detailed API documentation:

```python
from training import framework, sb3_trainer, experiment, metrics, utils

help(framework.TrainerConfig)
help(sb3_trainer.SB3Trainer)
help(experiment.MultiAlgorithmBenchmark)
help(metrics.BalatroMetrics)
help(utils.create_balatro_env)
```

## Contributing

To improve the framework:

1. Follow existing code style (docstrings, type hints)
2. Add tests for new functionality
3. Update documentation
4. Submit PR with clear description

## License

Same as main project (MIT)
