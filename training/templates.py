"""
Experiment Templates
====================

Pre-configured experiment workflows for common training scenarios.

Features:
- Curriculum learning template
- Imitation learning template
- Multi-seed template for robustness
- Custom template creation
- Template composition and extension

Expected Impact: 50% faster experiment setup
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
from pathlib import Path

from training.experiment import ExperimentConfig


@dataclass
class TemplateConfig:
    """Base template configuration"""
    name: str
    description: str
    algorithm: str = "PPO"
    base_timesteps: int = 100_000
    seed: int = 42


class ExperimentTemplate(ABC):
    """Base class for experiment templates"""

    def __init__(self, config: TemplateConfig):
        """
        Initialize template

        Args:
            config: Template configuration
        """
        self.config = config
        self.experiments: List[ExperimentConfig] = []

    @abstractmethod
    def generate_experiments(self, n_runs: int = 1) -> List[ExperimentConfig]:
        """
        Generate experiment configurations

        Args:
            n_runs: Number of experiment runs to generate

        Returns:
            List of ExperimentConfig objects
        """
        pass

    def add_experiment(self, config: ExperimentConfig):
        """Add an experiment"""
        self.experiments.append(config)

    def get_experiments(self) -> List[ExperimentConfig]:
        """Get all generated experiments"""
        return self.experiments.copy()

    def save_to_file(self, path: Path):
        """Save experiments to JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'template': self.config.name,
            'n_experiments': len(self.experiments),
            'experiments': [e.to_dict() for e in self.experiments]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(self.experiments)} experiments to {path}")


class CurriculumLearningTemplate(ExperimentTemplate):
    """
    Curriculum learning template

    Gradually increases difficulty by limiting max ante.
    """

    def __init__(self,
                 algorithm: str = "PPO",
                 base_timesteps: int = 100_000,
                 seed: int = 42):
        """Initialize curriculum learning template"""
        config = TemplateConfig(
            name="curriculum_learning",
            description="Gradually increase difficulty from ante 1 to 8",
            algorithm=algorithm,
            base_timesteps=base_timesteps,
            seed=seed
        )
        super().__init__(config)

        self.ante_levels = [1, 2, 3, 4, 5, 6, 7, 8]
        self.success_threshold = 0.8
        self.eval_interval = 100

    def set_ante_levels(self, levels: List[int]) -> "CurriculumLearningTemplate":
        """Set custom ante levels"""
        self.ante_levels = levels
        return self

    def set_success_threshold(self, threshold: float) -> "CurriculumLearningTemplate":
        """Set success threshold for advancing"""
        self.success_threshold = threshold
        return self

    def generate_experiments(self, n_runs: int = 1) -> List[ExperimentConfig]:
        """Generate curriculum learning experiments"""
        self.experiments = []

        for run in range(n_runs):
            for level, ante in enumerate(self.ante_levels):
                config = ExperimentConfig(
                    name=f"curriculum_level{level}_ante{ante}_run{run}",
                    algorithm=self.config.algorithm,
                    total_timesteps=self.config.base_timesteps,
                    seed=self.config.seed + run,
                    use_curriculum=True,
                    hyperparams_override={
                        'curriculum_max_ante': ante,
                        'curriculum_success_threshold': self.success_threshold,
                    }
                )
                self.experiments.append(config)

        print(f"Generated {len(self.experiments)} curriculum learning experiments")
        return self.experiments


class ImitationLearningTemplate(ExperimentTemplate):
    """
    Imitation learning template

    Pre-train with behavioral cloning then fine-tune with RL.
    """

    def __init__(self,
                 algorithm: str = "PPO",
                 base_timesteps: int = 100_000,
                 bc_timesteps: int = 10_000,
                 seed: int = 42):
        """Initialize imitation learning template"""
        config = TemplateConfig(
            name="imitation_learning",
            description="Pre-train with behavioral cloning, then fine-tune with RL",
            algorithm=algorithm,
            base_timesteps=base_timesteps,
            seed=seed
        )
        super().__init__(config)
        self.bc_timesteps = bc_timesteps

    def generate_experiments(self, n_runs: int = 1) -> List[ExperimentConfig]:
        """Generate imitation learning experiments"""
        self.experiments = []

        for run in range(n_runs):
            config = ExperimentConfig(
                name=f"imitation_learning_run{run}",
                algorithm=self.config.algorithm,
                total_timesteps=self.config.base_timesteps,
                seed=self.config.seed + run,
                use_behavioral_cloning=True,
                hyperparams_override={
                    'bc_timesteps': self.bc_timesteps,
                    'bc_learning_rate': 1e-3,
                }
            )
            self.experiments.append(config)

        print(f"Generated {len(self.experiments)} imitation learning experiments")
        return self.experiments


class MultiSeedTemplate(ExperimentTemplate):
    """
    Multi-seed template

    Run same configuration with multiple seeds for robustness analysis.
    """

    def __init__(self,
                 algorithm: str = "PPO",
                 base_timesteps: int = 100_000,
                 base_seed: int = 42):
        """Initialize multi-seed template"""
        config = TemplateConfig(
            name="multi_seed",
            description="Run multiple seeds for robustness",
            algorithm=algorithm,
            base_timesteps=base_timesteps,
            seed=base_seed
        )
        super().__init__(config)

    def generate_experiments(self, n_runs: int = 5) -> List[ExperimentConfig]:
        """Generate multi-seed experiments"""
        self.experiments = []

        for run in range(n_runs):
            config = ExperimentConfig(
                name=f"seed_run{run}",
                algorithm=self.config.algorithm,
                total_timesteps=self.config.base_timesteps,
                seed=self.config.seed + run,
            )
            self.experiments.append(config)

        print(f"Generated {n_runs} experiments with different seeds")
        return self.experiments


class HyperparameterVariationTemplate(ExperimentTemplate):
    """
    Hyperparameter variation template

    Test different hyperparameter values systematically.
    """

    def __init__(self,
                 algorithm: str = "PPO",
                 base_timesteps: int = 100_000,
                 seed: int = 42):
        """Initialize hyperparameter variation template"""
        config = TemplateConfig(
            name="hyperparameter_variation",
            description="Test different hyperparameter combinations",
            algorithm=algorithm,
            base_timesteps=base_timesteps,
            seed=seed
        )
        super().__init__(config)
        self.variations: Dict[str, List[Any]] = {}

    def add_variation(self, param_name: str, values: List[Any]) -> "HyperparameterVariationTemplate":
        """Add hyperparameter variation"""
        self.variations[param_name] = values
        return self

    def generate_experiments(self, n_runs: int = 1) -> List[ExperimentConfig]:
        """Generate hyperparameter variation experiments"""
        self.experiments = []

        if not self.variations:
            print("No hyperparameter variations defined")
            return self.experiments

        # Generate cartesian product of all variations
        import itertools

        param_names = list(self.variations.keys())
        param_values = [self.variations[name] for name in param_names]

        run_id = 0
        for combo in itertools.product(*param_values):
            for run in range(n_runs):
                hp_override = {}
                for name, value in zip(param_names, combo):
                    hp_override[name] = value

                config = ExperimentConfig(
                    name=f"hpvar_run{run_id}",
                    algorithm=self.config.algorithm,
                    total_timesteps=self.config.base_timesteps,
                    seed=self.config.seed + run,
                    hyperparams_override=hp_override
                )
                self.experiments.append(config)
                run_id += 1

        print(f"Generated {len(self.experiments)} hyperparameter variation experiments")
        return self.experiments


class AlgorithmComparisonTemplate(ExperimentTemplate):
    """
    Algorithm comparison template

    Compare multiple algorithms on equal footing.
    """

    def __init__(self,
                 algorithms: Optional[List[str]] = None,
                 base_timesteps: int = 100_000,
                 seed: int = 42):
        """Initialize algorithm comparison template"""
        if algorithms is None:
            algorithms = ["PPO", "A2C", "DQN"]

        config = TemplateConfig(
            name="algorithm_comparison",
            description=f"Compare algorithms: {', '.join(algorithms)}",
            algorithm="mixed",
            base_timesteps=base_timesteps,
            seed=seed
        )
        super().__init__(config)
        self.algorithms = algorithms

    def generate_experiments(self, n_runs: int = 1) -> List[ExperimentConfig]:
        """Generate algorithm comparison experiments"""
        self.experiments = []

        for algo in self.algorithms:
            for run in range(n_runs):
                config = ExperimentConfig(
                    name=f"{algo}_run{run}",
                    algorithm=algo,
                    total_timesteps=self.config.base_timesteps,
                    seed=self.config.seed + run,
                )
                self.experiments.append(config)

        print(f"Generated {len(self.experiments)} algorithm comparison experiments")
        return self.experiments


class TemplateRegistry:
    """Registry of available templates"""

    _templates = {
        'curriculum': CurriculumLearningTemplate,
        'imitation': ImitationLearningTemplate,
        'multi_seed': MultiSeedTemplate,
        'hyperparameter': HyperparameterVariationTemplate,
        'comparison': AlgorithmComparisonTemplate,
    }

    @classmethod
    def get_template(cls, name: str, **kwargs) -> ExperimentTemplate:
        """Get a template by name"""
        if name not in cls._templates:
            raise ValueError(f"Unknown template: {name}")

        template_class = cls._templates[name]
        return template_class(**kwargs)

    @classmethod
    def list_templates(cls) -> List[str]:
        """List available templates"""
        return list(cls._templates.keys())

    @classmethod
    def register_template(cls, name: str, template_class: type):
        """Register a custom template"""
        cls._templates[name] = template_class


# Convenience functions
def curriculum_learning(algorithm: str = "PPO",
                       timesteps: int = 100_000,
                       seed: int = 42) -> CurriculumLearningTemplate:
    """Create curriculum learning template"""
    return CurriculumLearningTemplate(algorithm, timesteps, seed)


def imitation_learning(algorithm: str = "PPO",
                      timesteps: int = 100_000,
                      bc_timesteps: int = 10_000,
                      seed: int = 42) -> ImitationLearningTemplate:
    """Create imitation learning template"""
    return ImitationLearningTemplate(algorithm, timesteps, bc_timesteps, seed)


def multi_seed(algorithm: str = "PPO",
              timesteps: int = 100_000,
              seed: int = 42) -> MultiSeedTemplate:
    """Create multi-seed template"""
    return MultiSeedTemplate(algorithm, timesteps, seed)


def algorithm_comparison(timesteps: int = 100_000,
                        algorithms: Optional[List[str]] = None,
                        seed: int = 42) -> AlgorithmComparisonTemplate:
    """Create algorithm comparison template"""
    return AlgorithmComparisonTemplate(algorithms, timesteps, seed)
