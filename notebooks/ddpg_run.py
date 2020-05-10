import gym

from common.utils import mini_batch_train
from models.ddpg_agent import DDPGAgent

gym.envs.register(
     id='CustomEnv-v333',
     entry_point='envs.custom_env_dir:RobotEnv')

env = gym.make("CustomEnv-v333")

max_episodes = 5000
max_steps = 150
batch_size = 128

gamma = 0.97
tau = 1e-2
buffer_maxlen = 100000
critic_lr = 5e-4
actor_lr = 5e-4

agent = DDPGAgent(env, gamma, tau, buffer_maxlen, critic_lr, actor_lr)
episode_rewards = mini_batch_train(env, agent, max_episodes, max_steps, batch_size)