import torch
import math

from envs.ur3_simenv.fastrobot.robot import robot


def check_tcp_diff_for_success(tcp_diff):
    return True


class robot_mdp:

    def __init__(self):
        self.fast_bot = robot()
        self.tcp_target = torch.eye(4)
        self.current_tcp = self.fast_bot.tcp([0.00, 0.00, 0.00, 0.00, 0.00, 0.00])

    def update_angle(self, angle_tensor):
        self.current_tcp = self.fast_bot.update_joint_angles_by(angle_tensor)
        tcp_diff = self.tcp_target - self.current_tcp
        success = check_tcp_diff_for_success(tcp_diff)
        return {self.fast_bot.joint_angles, tcp_diff, success}

    def reset_robot(self):
        self.fast_bot = robot()
        self.current_tcp = self.fast_bot.tcp([0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
        self.generate_target()

    def generate_target(self):
        max = math.pi / 2
        min = 0
        target_angle = (max - min) * torch.rand((6)) + min
        fast_bot = robot()
        self.tcp_target = fast_bot.tcp(target_angle)
