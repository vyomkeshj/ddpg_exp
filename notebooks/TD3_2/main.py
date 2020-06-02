import os

import gym
import numpy as np
import torch

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
import TD3, replay_buffer
from experience_replay_buffer import HindsightExperienceReplayBuffer
from replay_buffer import ReplayBuffer


def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = env_name;
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over ", eval_episodes + " episodes: ", avg_reward)
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    policy = "TD3"
    seed = 0
    start_timesteps = 25e3
    eval_freq = 5e3
    max_timesteps = 4e5
    expl_noise = 0.1
    batch_size = 256
    discount = 0.99
    tau = 0.0005
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2

    gym.envs.register(
        id='CustomEnv-v666',
        entry_point='envs.ur3_simenv:fast_environment')

    env = gym.make("CustomEnv-v666")

    # Set seeds
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": discount,
        "tau": tau,
    }

    # Initialize policy
    if policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = policy_noise * max_action
        kwargs["noise_clip"] = noise_clip * max_action
        kwargs["policy_freq"] = policy_freq
        policy = TD3.TD3(**kwargs)
    # policy.load("./models/model-her")
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    # evaluations = [eval_policy(policy, env, seed)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action\
        next_state, reward, done, _ = env.step(action)
        
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        epoch_end = False if episode_timesteps < env._max_episode_steps else True
        # Store data in replay buffer,
        # todo do the HER modification here
        # print("adding state to replay buffer", state)
      
        replay_buffer.add(state, action, next_state, reward, done_bool)  # goes to temp buffer

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= start_timesteps:
            policy.train(replay_buffer, batch_size)

        if bool(done) | epoch_end:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print("Total T:", {t + 1}, " Episode Num:", {episode_num + 1}, " Episode T:", {episode_timesteps},
                  " Reward:", {episode_reward})
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            policy.save("./models/model-" + str(episode_num))
