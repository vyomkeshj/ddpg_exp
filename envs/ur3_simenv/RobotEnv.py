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
        self.action_space = spaces.Box(low=np.array([-4.0000, -4.0000]), high=np.array([4.0000, 4.0000]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-math.pi, -math.pi, -1, -1, -1]),
                                            high=np.array([math.pi, math.pi,  1, 1, 1]),
                                            dtype=np.float32)

    def step(self, action):
        #print("action command= %.10f, %.10f" % (action[0], action[1]))

        self.socket.send_string("%f,%f" % (action[0], action[1]))
        message = self.socket.recv().decode("utf-8")
        data = message.split(',');
        float_data = list(map(lambda x: float(x), data))
        obs = np.array(float_data[0:5])

        obs[0] = (3.14/180)*obs[0];
        obs[1] = (3.14/180)*obs[1];
        #obs[2] = (3.14/180)*obs[2];
        #obs[3] = (3.14/180)*obs[3];
        #obs[4] = (3.14/180)*obs[4];

        self.action_count = self.action_count+1;
        #print("count=", self.action_count)
        #print("action command= %.10f, %.10f" % (action[0], action[1]))
        #print("observation received=",obs)
        #print("done status=",float_data)

        return obs, float_data[5]*0.100, float_data[6], {}

    def reset(self):
        #print('Environment reset')
        self.socket.send_string("r")
        message = self.socket.recv().decode("utf-8")
        #print("Received respose: %s" % message)
        data = message.split(',');
        float_data = list(map(lambda x: float(x), data))
        obs = np.array(float_data[0:5])
        return obs

    def render(self, mode='human'):
        pass