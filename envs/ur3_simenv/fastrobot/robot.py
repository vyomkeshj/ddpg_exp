import torch
import math
from envs.ur3_simenv.fastrobot.torch_bot import torchbot


class robot(object):

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.joint_angles = torch.tensor([0.00, 0.00, 0.00, 0.00, 0.00, 0.00]).to(self.device)
        self.robot = torchbot()

        self.max_step = 5
        self.step_size = 10.0
        self.alpha = 0.5
        self.tcp = self.robot.get_tcp(self.joint_angles)

        self.momentum_step_size = torch.tensor([0.00, 0.00, 0.00, 0.00, 0.00, 0.00]).to(self.device)

    def update_joint_angles_by(self, angle_tensor):
        self.momentum_step_size = self.alpha * self.momentum_step_size + (1 - self.alpha) * angle_tensor
        self.fix_momentum_step_size()
        angle_tensor = self.joint_angles + self.momentum_step_size
        angle_tensor -= 360 * math.floor((angle_tensor + 180) / 360)  # todo: really?
        self.joint_angles = angle_tensor
        self.tcp = self.robot.get_tcp(self.joint_angles)

    def get_robot_tcp(self):
        return self.tcp

    def fix_momentum_step_size(self):
        if self.momentum_step_size < -self.max_step:
            self.momentum_step_size = -self.max_step
        elif self.momentum_step_size > self.max_step:
            self.momentum_step_size = self.max_step

        if self.momentum_step_size < -self.max_step:
            self.momentum_step_size = -self.max_step
        elif self.momentum_step_size > self.max_step:
            self.momentum_step_size = self.max_step
