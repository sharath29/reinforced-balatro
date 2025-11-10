#!/usr/bin/env python3
"""
Framework Validation Tests
==========================

Quick tests to ensure the framework components work correctly.
Run this before using the full training scripts.

Usage:
    python test_framework.py
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from training.framework import TrainerConfig, RLTrainer, create_trainer
        from training.sb3_trainer import SB3Trainer, BalatroFeaturesExtractor
        from training.experiment import ExperimentConfig, MultiAlgorithmBenchmark
        from training.metrics import BalatroMetrics
        from training.utils import create_balatro_env, make_vectorized_env, validate_environment
        from training.env_wrapper import BalatroGymWrapper, CurriculumWrapper
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config_creation():
    """Test TrainerConfig creation"""
    print("\nTesting configuration...")
    try:
        from training.framework import TrainerConfig, HyperparameterConfig, get_default_config

        # Test default config
        config = get_default_config("PPO")
        assert config.algorithm == "PPO"
        assert config.total_timesteps == 1_000_000

        # Test custom config
        config = TrainerConfig(
            algorithm="A2C",
            total_timesteps=50_000,
            n_envs=2,
            seed=42
        )
        assert config.algorithm == "A2C"
        assert config.n_envs == 2

        # Test serialization
        config_dict = config.to_dict()
        config_loaded = TrainerConfig.from_dict(config_dict)
        assert config_loaded.algorithm == config.algorithm

        print("✓ Configuration tests passed")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment():
    """Test environment creation and validation"""
    print("\nTesting environment...")
    try:
        from training.utils import create_balatro_env, validate_environment

        # Create environment
        env = create_balatro_env(seed=42)
        print("✓ Environment created")

        # Validate
        if not validate_environment(env):
            raise RuntimeError("Environment validation failed")

        # Test reset
        obs, info = env.reset()
        assert obs is not None
        print("✓ Environment reset works")

        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None
        print("✓ Environment step works")

        env.close()
        print("✓ Environment tests passed")
        return True
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_creation():
    """Test trainer creation"""
    print("\nTesting trainer creation...")
    try:
        from training.framework import get_default_config, create_trainer
        from training.utils import create_balatro_env

        env = create_balatro_env(seed=42)
        config = get_default_config("PPO")
        config.total_timesteps = 100  # Minimal

        trainer = create_trainer(env, config)
        assert trainer is not None
        assert trainer.config.algorithm == "PPO"
        print("✓ Trainer creation successful")

        trainer.close()
        return True
    except Exception as e:
        print(f"✗ Trainer creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_experiment_config():
    """Test experiment configuration"""
    print("\nTesting experiment config...")
    try:
        from training.experiment import ExperimentConfig

        config = ExperimentConfig(
            algorithm="PPO",
            total_timesteps=50_000,
            name="test_experiment"
        )

        # Convert to trainer config
        trainer_config = config.to_trainer_config()
        assert trainer_config.algorithm == "PPO"

        print("✓ Experiment config tests passed")
        return True
    except Exception as e:
        print(f"✗ Experiment config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test metrics collection"""
    print("\nTesting metrics...")
    try:
        from training.metrics import BalatroMetrics

        metrics = BalatroMetrics()
        metrics.add_episode(reward=100.0, ante=5, score=50000, length=500)
        metrics.add_episode(reward=150.0, ante=6, score=60000, length=600)

        summary = metrics.get_summary()
        assert summary['num_episodes'] == 2
        assert summary['mean_reward'] > 0
        assert summary['mean_ante'] > 0

        print("✓ Metrics tests passed")
        return True
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wrappers():
    """Test environment wrappers"""
    print("\nTesting environment wrappers...")
    try:
        from training.utils import create_balatro_env
        from training.env_wrapper import BalatroGymWrapper, CurriculumWrapper

        env = create_balatro_env(seed=42)

        # Test standard wrapper
        env = BalatroGymWrapper(env)
        obs, info = env.reset()
        assert obs is not None
        print("✓ BalatroGymWrapper works")

        env.close()

        # Test curriculum wrapper
        env = create_balatro_env(seed=42)
        env = CurriculumWrapper(env)
        obs, info = env.reset()
        assert obs is not None
        curriculum_info = env.get_curriculum_info()
        assert 'current_max_ante' in curriculum_info
        print("✓ CurriculumWrapper works")

        env.close()
        return True
    except Exception as e:
        print(f"✗ Wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("Framework Validation Tests")
    print("="*70)

    results = [
        ("Imports", test_imports()),
        ("Configuration", test_config_creation()),
        ("Environment", test_environment()),
        ("Trainer Creation", test_trainer_creation()),
        ("Experiment Config", test_experiment_config()),
        ("Metrics", test_metrics()),
        ("Wrappers", test_wrappers()),
    ]

    # Print summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<30} {status}")

    print("="*70)
    print(f"Result: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! Framework is ready to use.")
        print("\nNext steps:")
        print("  1. Run quick test: python quick_train.py --quick-test")
        print("  2. Compare algorithms: python compare_algorithms.py --quick")
        print("  3. Full training: python train_with_framework.py")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please fix errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
