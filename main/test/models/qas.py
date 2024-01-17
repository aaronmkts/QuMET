import os
import sys

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(
     os.path.join(
         os.path.dirname(os.path.realpath(__file__)), "..", "..", ".." ,"main"
     )
    )

import gymnasium
import torch.optim as optim
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from codebase.gyms import QuMETGymnasium

#from codebase.actions.rl_train import get_agent
from stable_baselines3.common.env_checker import check_env

# Parameters 
env_name = 'NoisyTwoQubitEnv'

config = {"fidelity_threshold": 0.95,
          "reward_penalty":0.01,
          "max_timesteps": 20
}
# Environment

environment = QuMETGymnasium(env_name, config).make_gym()
print(environment)
''' 
#env = gym3.ToGymEnv(env)

#for idx, gate in enumerate(env.action_gates):
#    print('Action({:02d}) --> {}'.format(idx, gate))

#for idx, observable in enumerate(env.state_observables):
#    print('State({:02d}) --> {}'.format(idx, observable))


# Parameters
gamma = 0.99
learning_rate = 0.0001
policy_kwargs = dict(optimizer_class=optim.Adam)
agent = "a2c"
policy = "MlpPolicy"

#get_agent(agent_type, policy, env, gamma, learning_rate, policy_kwargs)

a2c_model = A2C(policy,
                        environment,
                        gamma=gamma,
                        learning_rate=learning_rate,
                        policy_kwargs=policy_kwargs,
                        tensorboard_log='logs/')
                        '''