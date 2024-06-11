import os
import sys

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(
     os.path.join(
         os.path.dirname(os.path.realpath(__file__)), "..", "..", ".." ,"src"
     )
    )

import gymnasium
import torch.optim as optim
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from qumet.gyms import QuMETGymnasium
from qumet.tools import load_config
from qumet.actions.rl_train import train
from stable_baselines3.common.env_checker import check_env

def main():
    
    config = load_config("configs/by_environment/noiseless_envs/noiseless_two_qubit.toml")
    env_name = 'NoisyTwoQubitEnv'
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    gamma: float = 0.99
    agent_type: str = "a2c"
    policy: str = "MlpPolicy"
    total_timesteps: int = 10000
    save_path: str = "./ckpts/test"
    load_name: str = None
    load_type: str = ""
    environment = QuMETGymnasium(env_name, config).make_gym()

    train(
        optimizer=optimizer,
        learning_rate=learning_rate,
        gamma=gamma,
        agent_type=agent_type,
        policy=policy,
        total_timesteps=total_timesteps,
        save_path=save_path,
        load_name=load_name,
        load_type=load_type,
        environment=environment
    )

if __name__ == "__main__":
    main()
