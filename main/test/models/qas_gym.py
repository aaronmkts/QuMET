import gym
import torch.optim as optim
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from codebase.gyms.environments.basic_envs import *

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
