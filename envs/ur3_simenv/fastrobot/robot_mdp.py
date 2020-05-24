import  torch
import math

class robot_mdp:

    def __init__(self):
        max = 3.14
        min = -3.14
        self.target = (max-min)*torch.rand((6)) + min

