import gym
from gym import spaces
import numpy as np
import math

from envs.ur3_simenv.fastrobot.robot_mdp import robot_mdp


class fast_environment(gym.Env):

    def __init__(self):
        self.action_count = 0;
        self._max_episode_steps = 100
        self.robot_mdp = robot_mdp()
        self.action_space = spaces.Box(low=np.array([-4.0000, -4.0000]), high=np.array([4.0000, 4.0000]),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-math.pi, -math.pi, -1, -1, -1]),
                                            high=np.array([math.pi, math.pi, 1, 1, 1]),
                                            dtype=np.float32)

    def step(self, action):
        # print("action command= %.10f, %.10f" % (action[0], action[1]))
        observation = self.robot_mdp.update_angle(action)
        # todo: obs, float_data[5] * 0.100, float_data[6]
        return observation, {}

    def reset(self):
        return self.robot_mdp.reset_robot()

    def render(self, mode='human'):
        pass
