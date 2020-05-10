import gym
from gym import spaces
import numpy as np
import zmq
import math

class RobotEnv(gym.Env):

    def __init__(self):
        self.action_count = 0;
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5555")
        self._max_episode_steps = 100
        print('Environment initialized')
        self.action_space = spaces.Box(low=np.array([-1.0000, -1.0000]), high=np.array([1.0000, 1.0000]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-math.pi, -math.pi]),
                                            high=np.array([math.pi, math.pi]),
                                            dtype=np.float32)

    def step(self, action):
        self.socket.send_string("%f,%f" % (action[0], action[1]))
        message = self.socket.recv().decode("utf-8")
        data = message.split(',');
        float_data = list(map(lambda x: float(x), data))
        obs = np.array(float_data[0:2])

        obs[0] = (3.14/180)*obs[0];
        obs[1] = (3.14/180)*obs[1];
        self.action_count = self.action_count+1;
        #print("count=", self.action_count)
        #print("action command= %.10f, %.10f" % (action[0], action[1]))
        #print("observation received=",obs/3.14)
        #print("reward received=",float_data[2]*0.1)
        #print("done status=",float_data[3])

        return obs/3.14, float_data[2]*0.1, float_data[3], {}

    def reset(self):
        #print('Environment reset')
        self.socket.send_string("r")
        message = self.socket.recv().decode("utf-8")
        #("Received respose: %s" % message)
        data = message.split(',');
        float_data = list(map(lambda x: float(x), data))
        obs = np.array(float_data[0:2])
        return obs

    def render(self, mode='human'):
        pass