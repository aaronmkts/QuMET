import gym
from gym.envs.registration import register
from .environments import(is_noiseless_environment,
                          get_noiseless_environment_info,
                          get_noiseless_environment
    
)

from .utils import QumetEnvironmentInfo, EnvironmentType, EnvironmentTaskType


def get_environment_info(name: str) -> QumetEnvironmentInfo:
    if is_noiseless_environment(name):
        info = get_noiseless_environment_info(name)
    else:
        raise ValueError(f"Environment {name} not found")

    return info

def get_environment(
    name: str,
    task: str,
    dataset_info: dict = None,
):
    environment_info = get_environment_info(name)

    environment_kwargs = {
        "name": name,
        "task": task,
        "dataset_info": dataset_info,
    }

    match environment_info.environment_type:
        case EnvironmentType.NOISELESS:
            environment_cls = get_noiseless_environment(**environment_kwargs)
        case _:
            raise ValueError(f"Environment source {environment_info.environment_type} not supported")
        
    return environment_cls

class QuMETGymnasium():
    def __init__(
        self,
        name: str,
        config: dict,
    ) -> None:
        
        self.name = name
        self.config = config
        self.environment_info = get_environment_info(self.name)
        self.register_gym()

    def register_gym(self):
        
        match self.environment_info.task_type:
            case EnvironmentTaskType.STATE_PREPERATION:
                #Register the base class for task type
                register(id='QuantumArchSearchEnv',
                        entry_point='codebase.gyms.environments:QuantumArchSearchEnv',
                        nondeterministic=True)
                
                if self.environment_info.environment_type == EnvironmentType.NOISELESS:
                    #Register the base class for environment type
                    register(id='BasicNQubitEnv',
                            entry_point='codebase.gyms.environments.noiseless_envs:BasicNQubitEnv',
                            nondeterministic=True)
                    
                    if self.name == 'BasicNQubitEnv':
                        pass
                    else:
                        register(id=self.name,
                            entry_point=f'codebase.gyms.environments.noiseless_envs:{self.name}',
                            nondeterministic=True)
                        print(f'gyms.environments.noiseless_envs:{self.name}')
                else:
                    raise ValueError(f"Environment type {self.environment_info.environment_type} not supported")
                
            case _:
                raise ValueError(f"Environment task {self.environment_info.task_type} not supported")

    

    def make_gym(self):
        environment = gym.make(self.name, 
                               **self.config)
        
        return environment
    
    def make_environment(self):
        self.task = self.environment_info.task_type.value
         
        environment_cls = get_environment(name = self.name,
                                          task = self.task
                                      )
        environment = environment_cls(**self.config)
        
        return environment

