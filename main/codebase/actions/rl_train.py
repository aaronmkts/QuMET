import os
import sys

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(
     os.path.join(
         os.path.dirname(os.path.realpath(__file__)), "..", "..", ".." ,"main","codebase"
     )
    )

import gym3
import gym
import gymnasium
import torch.optim as optim
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from gyms.environments import *

def train_agent(agent_type, policy, env, gamma, learning_rate, policy_kwargs, n_epochs=4, clip_range=0.2):
    print("Start training")

    if agent_type == "ac2":
        # Agent
        a2c_model = A2C(policy,
                        env,
                        gamma=gamma,
                        learning_rate=learning_rate,
                        policy_kwargs=policy_kwargs,
                        tensorboard_log='logs/')
        
        try:
            a2c_model.learn(total_timesteps=20000)
            print("End training for A2C")
        except Exception as e:
            print(f"Training for A2C failed: {e}")
        
    elif agent_type == "ppo":
        # Agent
        ppo_model = PPO(policy,
                        env,
                        gamma=gamma,
                        n_epochs=n_epochs,
                        clip_range=clip_range,
                        learning_rate=learning_rate,
                        policy_kwargs=policy_kwargs,
                        tensorboard_log='logs/')

        try:
            ppo_model.learn(total_timesteps=20000)
            print("End training for PPO")
        except Exception as e:
            print(f"Training for PPO failed: {e}")
