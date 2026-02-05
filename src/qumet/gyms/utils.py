"""Environment utility classes and types for QuMET Gymnasium.

This module defines enums and dataclasses for environment metadata,
including environment types, task types, and environment information structures.
"""

from dataclasses import dataclass
from enum import Enum


class EnvironmentType(Enum):
    """Enumeration of quantum environment simulation types.

    Attributes:
        NOISELESS: Ideal quantum simulation without noise.
        NOISY: Quantum simulation with noise models.
    """

    NOISELESS = "noiseless"
    NOISY = "noisy"


class EnvironmentTaskType(Enum):
    """Enumeration of environment task types.

    Attributes:
        STATE_PREPERATION: Preparing quantum circuits for specific quantum states.
    """

    STATE_PREPERATION = "state_preperation"


@dataclass
class QumetEnvironmentInfo:
    """Environment metadata for QuMET Gymnasium environments.

    Attributes:
        name: Environment name.
        environment_type: Type of quantum simulation (NOISELESS or NOISY).
        task_type: Type of task the environment performs.
        quantum_state_preperation: Whether the environment is for quantum state preparation.
    """

    name: str
    environment_type: EnvironmentType
    task_type: EnvironmentTaskType
    quantum_state_preperation: bool = False

    def __post_init__(self):
        """Validate and convert environment configuration after initialization."""
        self.environment_type = (
            EnvironmentType(self.environment_type)
            if isinstance(self.environment_type, str)
            else self.environment_type
        )
        self.task_type = (
            EnvironmentTaskType(self.task_type)
            if isinstance(self.task_type, str)
            else self.task_type
        )

        if self.task_type == EnvironmentTaskType.STATE_PREPERATION:
            assert self.quantum_state_preperation

    @property
    def is_state_preperation_environment(self):
        """Check if the environment is for state preparation tasks.

        Returns:
            bool: True if the environment performs state preparation.
        """
        return self.task_type == EnvironmentTaskType.STATE_PREPERATION
