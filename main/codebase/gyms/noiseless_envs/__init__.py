from .noiseless_envs import *
from ..utils import QumetEnvironmentInfo

NOISELESS_ENVIRONMENTS = { 
    "NoiselessNQubitEnv":{
    "info": QumetEnvironmentInfo("NoiselessNQubitEnv", environment_type="noiseless", 
                                 task_type="state_preperation",
                                 quantum_state_preperation = True),
    "environment": NoiselessNQubitEnv    
},
    "NoiselessTwoQubitEnv":{
    "info": QumetEnvironmentInfo("NoislessTwoQubitEnv", environment_type="noiseless", 
                                 task_type="state_preperation",
                                 quantum_state_preperation = True),   
    "environment": NoiselessTwoQubitEnv
},
    "NoiselessThreeQubitEnv":{
    "info": QumetEnvironmentInfo("NoiselessThreeQubitEnv", environment_type="noiseless", 
                                 task_type="state_preperation",
                                 quantum_state_preperation = True),
    "environment": NoiselessThreeQubitEnv

},
}

def is_noiseless_environment(name: str) -> bool:
    return name in NOISELESS_ENVIRONMENTS

def get_noiseless_environment_info(name: str) -> QumetEnvironmentInfo:
    if name not in NOISELESS_ENVIRONMENTS:
        raise ValueError(f"Manual environment {name} is not supported")
    return NOISELESS_ENVIRONMENTS[name]["info"]

def get_noiseless_environment(
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
    if name not in NOISELESS_ENVIRONMENTS:
        raise ValueError(f"Manual environment {name} is not supported")
    environment_info: QumetEnvironmentInfo = NOISELESS_ENVIRONMENTS[name]["info"]
   
    if task in ['state_preperation']:
        assert(
            environment_info.quantum_state_preperation
        ), f"Task {task} is not supported for {name}"
    
        environment_cls = NOISELESS_ENVIRONMENTS[name]["environment"]
    else:
        raise ValueError(f"Task {task} is not supported for {name}")

    return environment_cls

