"""Noiseless quantum circuit environments for reinforcement learning.

This module provides ideal (noiseless) quantum simulation environments for
training quantum circuit architectures with reinforcement learning.
"""

from ..utils import QumetEnvironmentInfo
from .noiseless_envs import *

NOISELESS_ENVIRONMENTS = {
    "NoiselessNQubitEnv": {
        "info": QumetEnvironmentInfo(
            "NoiselessNQubitEnv",
            environment_type="noiseless",
            task_type="state_preperation",
            quantum_state_preperation=True,
        ),
        "environment": NoiselessNQubitEnv,
    },
    "NoiselessTwoQubitEnv": {
        "info": QumetEnvironmentInfo(
            "NoislessTwoQubitEnv",
            environment_type="noiseless",
            task_type="state_preperation",
            quantum_state_preperation=True,
        ),
        "environment": NoiselessTwoQubitEnv,
    },
    "NoiselessThreeQubitEnv": {
        "info": QumetEnvironmentInfo(
            "NoiselessThreeQubitEnv",
            environment_type="noiseless",
            task_type="state_preperation",
            quantum_state_preperation=True,
        ),
        "environment": NoiselessThreeQubitEnv,
    },
}


def is_noiseless_environment(name: str) -> bool:
    """Check if an environment name is a registered noiseless environment.

    Args:
        name: Environment name to check.

    Returns:
        bool: True if the environment is registered.
    """
    return name in NOISELESS_ENVIRONMENTS


def get_noiseless_environment_info(name: str) -> QumetEnvironmentInfo:
    """Get metadata for a noiseless environment.

    Args:
        name: Name of the noiseless environment.

    Returns:
        QumetEnvironmentInfo: Environment metadata and configuration.

    Raises:
        ValueError: If the environment is not supported.
    """
    if name not in NOISELESS_ENVIRONMENTS:
        raise ValueError(f"Manual environment {name} is not supported")
    return NOISELESS_ENVIRONMENTS[name]["info"]


def get_noiseless_environment(
    name: str,
    task: str,
    dataset_info: dict = None,
):
    """Get a noiseless environment class for a specific task.

    Args:
        name: Name of the noiseless environment.
        task: Task type (e.g., 'state_preperation').
        dataset_info: Optional dataset metadata for environment configuration.

    Returns:
        type: Environment class configured for the specified task.

    Raises:
        ValueError: If the environment or task is not supported.
        AssertionError: If the environment doesn't support the specified task.
    """
    if name not in NOISELESS_ENVIRONMENTS:
        raise ValueError(f"Manual environment {name} is not supported")
    environment_info: QumetEnvironmentInfo = NOISELESS_ENVIRONMENTS[name]["info"]

    if task in ["state_preperation"]:
        assert (
            environment_info.quantum_state_preperation
        ), f"Task {task} is not supported for {name}"

        environment_cls = NOISELESS_ENVIRONMENTS[name]["environment"]
    else:
        raise ValueError(f"Task {task} is not supported for {name}")

    return environment_cls
