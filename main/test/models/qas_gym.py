import os
import sys

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(
     os.path.join(
         os.path.dirname(os.path.realpath(__file__)), "..", "..", ".." ,"main","codebase"
     )
    )

import gym
import torch.optim as optim
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from gyms.environments import *

# Parameters 
env_name = 'BasicTwoQubit-v0'
fidelity_threshold = 0.95
reward_penalty = 0.01
max_timesteps = 20

# Environment
env = gym.make(env_name,
               fidelity_threshold=fidelity_threshold,
               reward_penalty=reward_penalty,
               max_timesteps=max_timesteps)

for idx, gate in enumerate(env.action_gates):
    print('Action({:02d}) --> {}'.format(idx, gate))

for idx, observable in enumerate(env.state_observables):
    print('State({:02d}) --> {}'.format(idx, observable))

# Parameters
gamma = 0.99
learning_rate = 0.0001
policy_kwargs = dict(optimizer_class=optim.Adam)

# Agent
a2c_model = A2C("MlpPolicy",
                env,
                gamma=gamma,
                learning_rate=learning_rate,
                policy_kwargs=policy_kwargs,
                tensorboard_log='logs/')