"""
Environment Wrappers and Enhancements
======================================

Provides wrappers and utilities for Balatro environment integration
with the training framework.

Key Classes:
- BalatroGymWrapper: Standardized wrapper for Balatro environments
- CurriculumWrapper: Curriculum learning support
"""

import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple
import numpy as np


class BalatroGymWrapper(gym.Wrapper):
    """
    Standardized wrapper for Balatro environments.

    Ensures consistent behavior across different environment implementations
    and provides metadata for algorithm selection.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        # Store environment metadata
        self.metadata.update({
            'observation_space_type': 'dict',
            'action_space_type': 'discrete',
            'game': 'balatro',
            'wrapped': True
        })

        # Validate environment
        self._validate_env()

        # Track episode stats
        self.episode_rewards = []
        self.episode_antes = []
        self.episode_scores = []
        self.current_episode_reward = 0.0
        self.current_episode_ante = 1

    def _validate_env(self):
        """Validate environment compatibility"""
        if not isinstance(self.env.observation_space, spaces.Dict):
            raise ValueError("Balatro environment must have Dict observation space")

        required_obs_keys = ['hand', 'joker_ids', 'chips_scored', 'chips_needed',
                            'progress_ratio', 'money', 'ante', 'round',
                            'hands_left', 'discards_left', 'hand_levels', 'phase']

        for key in required_obs_keys:
            if key not in self.env.observation_space.spaces:
                raise ValueError(f"Missing required observation: {key}")

    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset environment"""
        obs, info = self.env.reset(**kwargs)

        self.current_episode_reward = 0.0
        self.current_episode_ante = 1

        return obs, info

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """Step environment"""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Track current episode stats
        self.current_episode_reward += reward
        if 'ante' in info:
            self.current_episode_ante = info['ante']

        # Record episode completion
        if terminated or truncated:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_antes.append(self.current_episode_ante)
            if 'final_score' in info:
                self.episode_scores.append(info.get('final_score', 0))

        return obs, reward, terminated, truncated, info

    def get_episode_stats(self) -> Dict[str, Any]:
        """Get episode statistics"""
        if not self.episode_rewards:
            return {}

        return {
            'total_episodes': len(self.episode_rewards),
            'mean_reward': float(np.mean(self.episode_rewards)),
            'std_reward': float(np.std(self.episode_rewards)),
            'max_reward': float(np.max(self.episode_rewards)),
            'mean_ante': float(np.mean(self.episode_antes)),
            'max_ante': int(np.max(self.episode_antes)),
            'mean_score': float(np.mean(self.episode_scores)) if self.episode_scores else 0.0,
            'max_score': float(np.max(self.episode_scores)) if self.episode_scores else 0.0,
        }

    def reset_stats(self):
        """Reset collected statistics"""
        self.episode_rewards = []
        self.episode_antes = []
        self.episode_scores = []


class CurriculumWrapper(gym.Wrapper):
    """
    Implements curriculum learning for Balatro.

    Gradually increases difficulty by limiting maximum ante until
    agent reaches success threshold at current level.
    """

    def __init__(self,
                 env: gym.Env,
                 initial_max_ante: int = 2,
                 ante_increment: int = 1,
                 success_threshold: float = 0.8,
                 eval_interval: int = 100):
        """
        Initialize curriculum wrapper

        Args:
            env: Balatro environment
            initial_max_ante: Starting maximum ante
            ante_increment: How much to increase ante per level
            success_threshold: Success rate needed to advance (0-1)
            eval_interval: Episodes between difficulty checks
        """
        super().__init__(env)

        self.initial_max_ante = initial_max_ante
        self.current_max_ante = initial_max_ante
        self.ante_increment = ante_increment
        self.success_threshold = success_threshold
        self.eval_interval = eval_interval

        # Track performance
        self.episode_count = 0
        self.episode_antes = []
        self.curriculum_history = []

    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset environment"""
        obs, info = self.env.reset(**kwargs)
        self.episode_count += 1

        # Check if we should increase difficulty
        if self.episode_count % self.eval_interval == 0:
            self._check_difficulty_increase()

        return obs, info

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """Step environment"""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Track ante
        ante = info.get('ante', 1)
        self.episode_antes.append(ante)

        # Terminate episode if ante exceeds curriculum limit
        if ante > self.current_max_ante:
            terminated = True
            info['curriculum_limited'] = True

        return obs, reward, terminated, truncated, info

    def _check_difficulty_increase(self):
        """Check if we should increase difficulty"""
        if len(self.episode_antes) < self.eval_interval:
            return

        # Calculate success rate in last eval_interval episodes
        recent_antes = self.episode_antes[-self.eval_interval:]
        success_count = sum(1 for a in recent_antes if a >= self.current_max_ante)
        success_rate = success_count / self.eval_interval

        if success_rate >= self.success_threshold:
            old_max_ante = self.current_max_ante
            self.current_max_ante += self.ante_increment

            log_entry = {
                'episode': self.episode_count,
                'old_max_ante': old_max_ante,
                'new_max_ante': self.current_max_ante,
                'success_rate': success_rate,
            }
            self.curriculum_history.append(log_entry)

            print(f"[Curriculum] Episode {self.episode_count}: "
                  f"Difficulty increased from {old_max_ante} to {self.current_max_ante} "
                  f"(success rate: {success_rate:.1%})")

    def get_curriculum_info(self) -> Dict[str, Any]:
        """Get curriculum learning information"""
        return {
            'current_max_ante': self.current_max_ante,
            'episode_count': self.episode_count,
            'history': self.curriculum_history,
        }

    def reset_curriculum(self):
        """Reset curriculum to initial state"""
        self.current_max_ante = self.initial_max_ante
        self.episode_antes = []
        self.curriculum_history = []


class RewardScalingWrapper(gym.Wrapper):
    """Scale rewards to a specific range"""

    def __init__(self, env: gym.Env, scale: float = 1.0, offset: float = 0.0):
        """
        Initialize reward scaling

        Args:
            env: Environment to wrap
            scale: Scaling factor
            offset: Offset to add
        """
        super().__init__(env)
        self.scale = scale
        self.offset = offset

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """Step with scaled reward"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        scaled_reward = reward * self.scale + self.offset
        return obs, scaled_reward, terminated, truncated, info


class ActionMaskWrapper(gym.Wrapper):
    """Enforce action masking through error handling"""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.valid_action_mask = None

    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset environment"""
        obs, info = self.env.reset(**kwargs)

        # Update action mask if available
        if isinstance(obs, dict) and 'action_mask' in obs:
            self.valid_action_mask = obs['action_mask']

        return obs, info

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """Step with action validation"""
        # Validate action if mask available
        if self.valid_action_mask is not None:
            if not self.valid_action_mask[action]:
                print(f"Warning: Invalid action {action} selected. "
                      "Consider using action masking in your policy.")

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Update action mask
        if isinstance(obs, dict) and 'action_mask' in obs:
            self.valid_action_mask = obs['action_mask']

        return obs, reward, terminated, truncated, info


def wrap_balatro_env(env: gym.Env,
                     use_curriculum: bool = False,
                     curriculum_config: Optional[Dict[str, Any]] = None) -> gym.Env:
    """
    Wrap a Balatro environment with standard wrappers

    Args:
        env: Base Balatro environment
        use_curriculum: Whether to apply curriculum learning
        curriculum_config: Configuration for curriculum learning

    Returns:
        Wrapped environment
    """

    # Apply standard wrapper
    env = BalatroGymWrapper(env)

    # Apply curriculum if requested
    if use_curriculum:
        curriculum_config = curriculum_config or {}
        env = CurriculumWrapper(env, **curriculum_config)

    return env
