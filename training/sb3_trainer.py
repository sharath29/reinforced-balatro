"""
Stable-Baselines3 Trainer Implementation
=========================================

Implements the RLTrainer interface for Stable-Baselines3 algorithms.
Supports PPO, A2C, and DQN with unified hyperparameter management.
"""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.ppo import MlpPolicy

from training.framework import RLTrainer, TrainerConfig


# ---------------------------------------------------------------------------
# Custom Feature Extractor for Balatro
# ---------------------------------------------------------------------------

class BalatroFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that understands Balatro's observation structure.

    Processes structured observations with:
    - Hand: 8x52 one-hot encoded cards
    - Jokers: 10x16 joker embeddings
    - Game state: Scalar features (chips, money, ante, etc.)
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        # Calculate input dimensions for each component
        hand_dim = 8 * 52  # One-hot encoded cards
        joker_dim = 10 * 16  # Joker embeddings
        game_state_dim = 32  # All scalar features

        # Build the network
        self.hand_net = nn.Sequential(
            nn.Linear(hand_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.joker_net = nn.Sequential(
            nn.Linear(joker_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.game_state_net = nn.Sequential(
            nn.Linear(game_state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Combine all features
        combined_dim = 128 + 64 + 32  # From each subnet
        self.combined_net = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Process hand (convert to one-hot)
        hand = observations['hand'].long()
        batch_size = hand.shape[0]
        hand_one_hot = torch.zeros(batch_size, 8, 52, device=hand.device)

        for i in range(8):
            valid_cards = hand[:, i] >= 0
            if valid_cards.any():
                hand_one_hot[valid_cards, i, hand[valid_cards, i]] = 1

        hand_features = self.hand_net(hand_one_hot.view(batch_size, -1))

        # Process jokers (simple embedding)
        joker_ids = observations['joker_ids'].float()
        joker_features = self.joker_net(joker_ids.view(batch_size, -1))

        # Process game state
        game_features = torch.cat([
            observations['chips_scored'].float().unsqueeze(1) / 1e6,  # Normalize
            observations['chips_needed'].float().unsqueeze(1) / 1e5,
            observations['progress_ratio'].float().unsqueeze(1),
            observations['money'].float().unsqueeze(1) / 100,
            observations['ante'].float().unsqueeze(1) / 10,
            observations['round'].float().unsqueeze(1) / 3,
            observations['hands_left'].float().unsqueeze(1) / 10,
            observations['discards_left'].float().unsqueeze(1) / 5,
            observations['hand_levels'].float() / 10,  # 12 values
            observations['phase'].float().unsqueeze(1) / 3,
        ], dim=1)

        game_state_features = self.game_state_net(game_features)

        # Combine all features
        combined = torch.cat([hand_features, joker_features, game_state_features], dim=1)
        return self.combined_net(combined)


# ---------------------------------------------------------------------------
# Custom Callbacks
# ---------------------------------------------------------------------------

class BalatroMetricsCallback(BaseCallback):
    """Track Balatro-specific metrics during training"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_antes = []
        self.episode_scores = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Check for episode end
        if len(self.locals.get('dones', [])) == 0:
            return True

        for i, done in enumerate(self.locals.get('dones', [])):
            if done:
                info = self.locals.get('infos', [{}])[i]

                # Extract metrics
                reward = self.locals.get('rewards', [0])[i]
                self.episode_rewards.append(reward)
                self.episode_antes.append(info.get('ante', 1))
                self.episode_scores.append(info.get('final_score', 0))
                self.episode_lengths.append(info.get('episode_length', 0))

                # Log to logger
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.record('balatro/episode_ante', info.get('ante', 1))
                    self.logger.record('balatro/episode_score', info.get('final_score', 0))
                    self.logger.record('balatro/episode_reward', reward)

        return True


# ---------------------------------------------------------------------------
# SB3 Trainer Implementation
# ---------------------------------------------------------------------------

class SB3Trainer(RLTrainer):
    """
    Stable-Baselines3 trainer implementation.

    Wraps PPO, A2C, or DQN algorithms with unified interface.
    """

    def __init__(self, env: gym.Env, config: TrainerConfig):
        """
        Initialize SB3Trainer

        Args:
            env: Gymnasium environment or vectorized environment
            config: TrainerConfig instance
        """
        super().__init__(env, config)

        # Prepare environment
        self.env = self._prepare_environment()

        # Create model
        self.model = self.create_model()

        # Callback for tracking metrics
        self.metrics_callback = BalatroMetricsCallback()

    def _prepare_environment(self) -> gym.Env:
        """Prepare environment with wrappers"""

        # Create vectorized environment if needed
        if not hasattr(self.env, 'num_envs'):
            # Single environment, wrap it
            env = self.env
        else:
            # Already vectorized
            return self.env

        # Apply wrappers
        if self.config.normalize_obs or self.config.normalize_reward:
            env = VecNormalize(
                self.env,
                norm_obs=self.config.normalize_obs,
                norm_reward=self.config.normalize_reward,
                clip_obs=10.0 if self.config.normalize_obs else 1.0
            )

        return env

    def _get_policy_kwargs(self) -> Dict[str, Any]:
        """Get policy kwargs for the algorithm"""

        policy_kwargs = {}

        # Add feature extractor if configured
        if self.config.use_feature_extractor and self.config.feature_extractor_type == "balatro":
            policy_kwargs['features_extractor_class'] = BalatroFeaturesExtractor
            policy_kwargs['features_extractor_kwargs'] = {
                'features_dim': self.config.hyperparams.features_dim
            }

        # Add network architecture
        if self.config.algorithm == "DQN":
            # DQN uses single net_arch
            policy_kwargs['net_arch'] = self.config.hyperparams.policy_net_arch
        else:
            # PPO and A2C use separate pi and vf
            policy_kwargs['net_arch'] = {
                'pi': self.config.hyperparams.policy_net_arch,
                'vf': self.config.hyperparams.value_net_arch
            }

        return policy_kwargs

    def _get_algorithm_kwargs(self) -> Dict[str, Any]:
        """Get algorithm-specific hyperparameters"""

        hp = self.config.hyperparams
        kwargs = {
            'learning_rate': hp.learning_rate,
            'gamma': hp.gamma,
        }

        algo_specific = hp.algorithm_specific

        if self.config.algorithm == "PPO":
            kwargs.update({
                'n_steps': algo_specific.get('n_steps', 2048),
                'batch_size': algo_specific.get('batch_size', 64),
                'n_epochs': algo_specific.get('n_epochs', 10),
                'gae_lambda': hp.gae_lambda,
                'clip_range': algo_specific.get('clip_range', 0.2),
                'ent_coef': hp.entropy_coefficient,
                'vf_coef': hp.value_function_coefficient,
                'max_grad_norm': hp.max_grad_norm,
            })

        elif self.config.algorithm == "A2C":
            kwargs.update({
                'n_steps': algo_specific.get('n_steps', 5),
                'gae_lambda': hp.gae_lambda,
                'ent_coef': hp.entropy_coefficient,
                'vf_coef': hp.value_function_coefficient,
                'max_grad_norm': hp.max_grad_norm,
            })

        elif self.config.algorithm == "DQN":
            kwargs.update({
                'buffer_size': algo_specific.get('buffer_size', 100_000),
                'learning_starts': algo_specific.get('learning_starts', 10_000),
                'batch_size': algo_specific.get('batch_size', 32),
                'train_freq': algo_specific.get('train_freq', 4),
                'target_update_interval': algo_specific.get('target_update_interval', 10_000),
                'exploration_fraction': algo_specific.get('exploration_fraction', 0.1),
                'exploration_initial_eps': algo_specific.get('exploration_initial_eps', 1.0),
                'exploration_final_eps': algo_specific.get('exploration_final_eps', 0.05),
            })

        return kwargs

    def create_model(self) -> Any:
        """Create the RL model"""

        policy_kwargs = self._get_policy_kwargs()
        algo_kwargs = self._get_algorithm_kwargs()

        # Add common kwargs
        algo_kwargs.update({
            'policy': 'MultiInputPolicy',
            'env': self.env,
            'verbose': 1,
            'tensorboard_log': str(self.save_path / "tb_logs"),
            'device': 'auto',
            'seed': self.config.seed,
            'policy_kwargs': policy_kwargs,
        })

        # Create model based on algorithm
        if self.config.algorithm == "PPO":
            model = PPO(**algo_kwargs)
        elif self.config.algorithm == "A2C":
            model = A2C(**algo_kwargs)
        elif self.config.algorithm == "DQN":
            model = DQN(**algo_kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")

        return model

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

        if self.model is None:
            raise RuntimeError("Model not created. Call create_model() first.")

        total_timesteps = total_timesteps or self.config.total_timesteps

        print(f"\n{'='*70}")
        print(f"Training {self.config.algorithm}")
        print(f"{'='*70}")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Number of environments: {getattr(self.env, 'num_envs', 1)}")
        print(f"Save directory: {self.save_path}")
        print(f"{'='*70}\n")

        # Create callbacks
        callbacks = []

        # Add provided callback
        if callback:
            callbacks.append(callback)

        # Add metrics callback
        callbacks.append(self.metrics_callback)

        # Add checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.checkpoint_freq,
            save_path=str(self.save_path / "checkpoints"),
            name_prefix=f"{self.config.algorithm}_checkpoint"
        )
        callbacks.append(checkpoint_callback)

        # Train the model
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=CallbackList(callbacks) if len(callbacks) > 1 else (callbacks[0] if callbacks else None),
                log_interval=self.config.log_interval,
                progress_bar=True
            )
            self.total_timesteps += total_timesteps
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")

        # Extract metrics
        self.metrics['episode_rewards'] = self.metrics_callback.episode_rewards
        self.metrics['episode_antes'] = self.metrics_callback.episode_antes
        self.metrics['episode_scores'] = self.metrics_callback.episode_scores
        self.metrics['episode_lengths'] = self.metrics_callback.episode_lengths
        self.metrics['training_steps'].append(self.total_timesteps)

        return {
            'total_timesteps': self.total_timesteps,
            'mean_reward': np.mean(self.metrics['episode_rewards']) if self.metrics['episode_rewards'] else 0,
            'std_reward': np.std(self.metrics['episode_rewards']) if self.metrics['episode_rewards'] else 0,
        }

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

        if self.model is None:
            raise RuntimeError("Model not created. Train first.")

        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.env,
            n_eval_episodes=n_episodes,
            deterministic=deterministic
        )

        self.metrics['mean_reward'].append(mean_reward)
        self.metrics['std_reward'].append(std_reward)

        return mean_reward, std_reward

    def close(self):
        """Close environment and cleanup"""
        if hasattr(self.env, 'close'):
            self.env.close()
        super().close()
