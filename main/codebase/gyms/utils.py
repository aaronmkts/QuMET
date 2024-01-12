from dataclasses import dataclass
from enum import Enum


class EnvironmentType(Enum):
    """
    The type of environment, must be one of the following:
    - NOISELESS: 
    """

    NOISELESS = "noiseless"
    NOISY = "noisy"


class EnvironmentTaskType(Enum):
    """
    The task type of the Environment, must be one of the following:
    STATE_PREPERATION: Preparing QC's for specific quantum states
    """

    STATE_PREPERATION = "state_preperation"
    # unitary preperation
    # VQE (hamiltonian ground states)


@dataclass
class QumetEnvironmentInfo:
    """
    The environment info for QuMET.
    """

    # environment name
    name: str

    environment_type: EnvironmentType
    task_type: EnvironmentTaskType

   
    # STATE PREPERATION environemnts
    quantum_state_preperation: bool = False



    def __post_init__(self):
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

        # State Preperation
        if self.task_type == EnvironmentTaskType.STATE_PREPERATION:
            assert self.quantum_state_preperation

    

    @property
    def is_state_preperation_environment(self):
        return self.task_type == EnvironmentTaskType.STATE_PREPERATION
