from notebooks.TD3.td3 import TD3Agent
import gym


gym.envs.register(
     id='CustomEnv-v333',
     entry_point='envs.custom_env_dir:RobotEnv')

env = gym.make("CustomEnv-v333")

gamma = 0.98
tau = 1e-2
noise_std = 0.7
bound = 0.5
delay_step = 2
buffer_maxlen = 25000
critic_lr = 1e-1
actor_lr = 1e-1

max_episodes = 10000
max_steps = 100
batch_size = 64

gym.envs.register(
     id='CustomEnv-v333',
     entry_point='envs.custom_env_dir:RobotEnv')

env = gym.make("CustomEnv-v333")

agent = TD3Agent(env, gamma, tau, buffer_maxlen, delay_step, noise_std, bound, critic_lr, actor_lr)

def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)

            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state

    return episode_rewards

episode_rewards = mini_batch_train(env, agent, max_episodes, max_steps, batch_size)