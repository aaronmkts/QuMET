from .noisy_envs import *
from ..utils import QumetEnvironmentInfo

NOISY_ENVIRONMENTS = { 
    "NoisyNQubitEnv":{
    "info": QumetEnvironmentInfo("NoisyNQubitEnv", environment_type="noisy", 
                                 task_type="state_preperation",
                                 quantum_state_preperation = True),
    "environment": NoisyNQubitEnv    
},
    "NoisyTwoQubitEnv":{
    "info": QumetEnvironmentInfo("NoisyTwoQubitEnv", environment_type="noisy", 
                                 task_type="state_preperation",
                                 quantum_state_preperation = True),   
    "environment": NoisyTwoQubitEnv
},
    "NoisyThreeQubitEnv":{
    "info": QumetEnvironmentInfo("NoisyThreeQubitEnv", environment_type="noisy", 
                                 task_type="state_preperation",
                                 quantum_state_preperation = True),
    "environment": NoisyThreeQubitEnv

},
}

def is_noisy_environment(name: str) -> bool:
    return name in NOISY_ENVIRONMENTS

def get_noisy_environment_info(name: str) -> QumetEnvironmentInfo:
    if name not in NOISY_ENVIRONMENTS:
        raise ValueError(f"Manual environment {name} is not supported")
    return NOISY_ENVIRONMENTS[name]["info"]

def get_noisy_environment(
    name: str,
    task: str,
    dataset_info: dict = None,
):
    """
    Args:
        name: The name of the environment.
        task: The task type.
        dataset_info: The dataset info.
    """
    if name not in NOISY_ENVIRONMENTS:
        raise ValueError(f"Manual environment {name} is not supported")
    environment_info: QumetEnvironmentInfo = NOISY_ENVIRONMENTS[name]["info"]
   
    if task in ['state_preperation']:
        assert(
            environment_info.quantum_state_preperation
        ), f"Task {task} is not supported for {name}"
    
        environment_cls = NOISY_ENVIRONMENTS[name]["environment"]
    else:
        raise ValueError(f"Task {task} is not supported for {name}")

    return environment_cls