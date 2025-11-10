"""
Training Utilities
==================

Helper functions for environment setup, logging, and configuration.
"""

from pathlib import Path
from typing import Callable, Optional, List, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor


def create_balatro_env(seed: int = None) -> gym.Env:
    """
    Create a Balatro environment

    Args:
        seed: Random seed for the environment

    Returns:
        Balatro environment instance
    """
    from balatro_gym.balatro_env_2 import BalatroEnv

    env = BalatroEnv(seed=seed)
    return env


def make_vectorized_env(env_creator: Callable,
                        n_envs: int = 4,
                        seed: int = 42,
                        use_subprocess: bool = True) -> gym.Env:
    """
    Create a vectorized environment

    Args:
        env_creator: Callable that creates a new environment
        n_envs: Number of parallel environments
        seed: Base seed (actual seeds will be seed + rank)
        use_subprocess: Whether to use SubprocVecEnv (True) or DummyVecEnv (False)

    Returns:
        Vectorized environment
    """

    def make_env(rank: int, seed: int = 0):
        def _init():
            env = env_creator()
            if hasattr(env, 'reset'):
                env.reset(seed=seed + rank)
            return Monitor(env)

        return _init

    if n_envs > 1 and use_subprocess:
        env = SubprocVecEnv([make_env(i, seed) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(i, seed) for i in range(n_envs)])

    return env


def setup_logging(save_dir: Path, algorithm: str):
    """Setup logging infrastructure"""
    import logging

    log_dir = save_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(algorithm)
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_dir / f"{algorithm}.log")
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def print_config(config: Any):
    """Pretty print configuration"""
    print("\n" + "=" * 70)
    print("Training Configuration")
    print("=" * 70)

    config_dict = config.to_dict() if hasattr(config, 'to_dict') else vars(config)

    def print_dict(d: Dict, indent: int = 0):
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                print_dict(value, indent + 1)
            elif isinstance(value, list):
                print("  " * indent + f"{key}: {value}")
            else:
                print("  " * indent + f"{key}: {value}")

    print_dict(config_dict)
    print("=" * 70 + "\n")


def validate_environment(env: gym.Env) -> bool:
    """Validate that environment is properly configured"""

    if not hasattr(env, 'observation_space'):
        print("Error: Environment missing observation_space")
        return False

    if not hasattr(env, 'action_space'):
        print("Error: Environment missing action_space")
        return False

    # Try reset
    try:
        obs, info = env.reset()
        if obs is None:
            print("Error: reset() returned None observation")
            return False
    except Exception as e:
        print(f"Error: reset() failed: {e}")
        return False

    # Try step
    try:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if obs is None:
            print("Error: step() returned None observation")
            return False
    except Exception as e:
        print(f"Error: step() failed: {e}")
        return False

    print("✓ Environment validation passed")
    return True


class EnvironmentWrapper(gym.Wrapper):
    """Base wrapper for custom environment modifications"""

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


class NormalizeReward(EnvironmentWrapper):
    """Normalize rewards to [-1, 1]"""

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        self.return_rms = RunningMeanStd(shape=())

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if isinstance(reward, (int, float)):
            self.return_rms.update(np.array([reward]))
            normalized_reward = reward / np.sqrt(self.return_rms.var + self.epsilon)
        else:
            normalized_reward = reward
        return obs, normalized_reward, terminated, truncated, info


class RunningMeanStd:
    """Running mean and std calculation"""

    def __init__(self, shape=()):
        import numpy as np
        self.shape = shape
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 0

    def update(self, x):
        import numpy as np
        x = np.asarray(x)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        import numpy as np
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


def save_experiment_results(results: Dict[str, Any], save_dir: Path):
    """Save experiment results"""
    import json

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {save_dir / 'results.json'}")


def load_experiment_results(save_dir: Path) -> Dict[str, Any]:
    """Load experiment results"""
    import json

    with open(save_dir / "results.json", 'r') as f:
        return json.load(f)
