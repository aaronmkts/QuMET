import os
import gym3
import gym
import gymnasium
import torch.optim as optim
import torch
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy


def get_agent(
    agent,
    policy,
    env,
    gamma,
    learning_rate,
    policy_kwargs,
    save_path,
    n_epochs=4,
    clip_range=0.2,
):

    match agent.lower():
        case "a2c":
            agent = A2C(
                policy,
                env,
                gamma=gamma,
                learning_rate=learning_rate,
                policy_kwargs=policy_kwargs,
                tensorboard_log=save_path,
            )

        case "ppo":
            agent = PPO(
                policy,
                env,
                gamma=gamma,
                n_epochs=n_epochs,
                clip_range=clip_range,
                learning_rate=learning_rate,
                policy_kwargs=policy_kwargs,
                tensorboard_log=save_path,
            )

        case _:
            raise ValueError(f"Unsupported agent: {agent.lower()}")

    return agent


def train(
    optimizer,
    learning_rate,
    gamma,
    agent_type,
    policy,
    total_timesteps,
    save_path,
    load_name,
    load_type,
    environment,
):

    if save_path is not None:
        # if save_path is None, the model will not be saved
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

    # opt = get_optimizer(optim.Adam,optimizer,learning_rate,0)

    policy_kwargs = dict(optimizer_class=optim.Adam)

    agent = get_agent(
        agent_type, policy, environment, gamma, learning_rate, policy_kwargs, save_path
    )

    # Training
    print("***** Running training *****")
    agent.learn(total_timesteps=total_timesteps)
    print("***** Training complete *****")
