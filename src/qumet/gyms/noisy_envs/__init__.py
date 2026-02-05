"""Noisy quantum circuit environments for reinforcement learning.

This module provides quantum simulation environments with noise models for
training robust quantum circuit architectures with reinforcement learning.
"""

from ..utils import QumetEnvironmentInfo
from .noisy_envs import *

NOISY_ENVIRONMENTS = {
    "NoisyNQubitEnv": {
        "info": QumetEnvironmentInfo(
            "NoisyNQubitEnv",
            environment_type="noisy",
            task_type="state_preperation",
            quantum_state_preperation=True,
        ),
        "environment": NoisyNQubitEnv,
    },
    "NoisyTwoQubitEnv": {
        "info": QumetEnvironmentInfo(
            "NoisyTwoQubitEnv",
            environment_type="noisy",
            task_type="state_preperation",
            quantum_state_preperation=True,
        ),
        "environment": NoisyTwoQubitEnv,
    },
    "NoisyThreeQubitEnv": {
        "info": QumetEnvironmentInfo(
            "NoisyThreeQubitEnv",
            environment_type="noisy",
            task_type="state_preperation",
            quantum_state_preperation=True,
        ),
        "environment": NoisyThreeQubitEnv,
    },
}


def is_noisy_environment(name: str) -> bool:
    """Check if an environment name is a registered noisy environment.
    
    Args:
        name: Environment name to check.
    
    Returns:
        bool: True if the environment is registered.
    """
    return name in NOISY_ENVIRONMENTS


def get_noisy_environment_info(name: str) -> QumetEnvironmentInfo:
    """Get metadata for a noisy environment.
    
    Args:
        name: Name of the noisy environment.
    
    Returns:
        QumetEnvironmentInfo: Environment metadata and configuration.
    
    Raises:
        ValueError: If the environment is not supported.
    """
    if name not in NOISY_ENVIRONMENTS:
        raise ValueError(f"Manual environment {name} is not supported")
    return NOISY_ENVIRONMENTS[name]["info"]


def get_noisy_environment(
    name: str,
    task: str,
    dataset_info: dict = None,
):
    """Get a noisy environment class for a specific task.
    
    Args:
        name: Name of the noisy environment.
        task: Task type (e.g., 'state_preperation').
        dataset_info: Optional dataset metadata for environment configuration.
    
    Returns:
        type: Environment class configured for the specified task.
    
    Raises:
        ValueError: If the environment or task is not supported.
        AssertionError: If the environment doesn't support the specified task.
    """
    if name not in NOISY_ENVIRONMENTS:
        raise ValueError(f"Manual environment {name} is not supported")
    environment_info: QumetEnvironmentInfo = NOISY_ENVIRONMENTS[name]["info"]

    if task in ["state_preperation"]:
        assert (
            environment_info.quantum_state_preperation
        ), f"Task {task} is not supported for {name}"

        environment_cls = NOISY_ENVIRONMENTS[name]["environment"]
    else:
        raise ValueError(f"Task {task} is not supported for {name}")

    return environment_cls
