import numpy as np
from math import *
import cmath
import torch


class torchbot:

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # d (unit: mm)
        self.d1 = 0.1519
        self.d2 = d3 = 0
        self.d4 = 0.11235
        self.d5 = 0.08535
        self.d6 = 0.0819

        # a (unit: mm)
        self.a1 = self.a4 = self.a5 = self.a6 = 0
        self.a2 = -0.24365
        self.a3 = -0.21325

        # List type of D-H parameter
        # Do not remove these
        self.d = np.array([self.d1, self.d2, self.d3, self.d4, self.d5, self.d6])  # unit: mm
        self.a = np.array([self.a1, self.a2, self.a3, self.a4, self.a5, self.a6])  # unit: mm
        self.alpha = np.array([pi / 2, 0, 0, pi / 2, -pi / 2, 0])  # unit: radian

    def tcp_internal(self, i, theta):
        """Calculate the tcp between two links.
        Args:
            i: A target index of joint value.
            theta: A list of joint value solution. (unit: radian)

        Returns:
            An tcp of Link l w.r.t. Link l-1, where l = i + 1.
        """
        Rot_z = torch.eye(4).to(self.device)
        Rot_z[0, 0] = Rot_z[1, 1] = cos(theta[i])
        Rot_z[0, 1] = -sin(theta[i])
        Rot_z[1, 0] = sin(theta[i])

        Trans_z = torch.eye(4).to(self.device)
        Trans_z[2, 3] = self.d[i]

        Trans_x = torch.eye(4).to(self.device)
        Trans_x[0, 3] = self.a[i]

        Rot_x = torch.eye(4).to(self.device)
        Rot_x[1, 1] = Rot_x[2, 2] = cos(self.alpha[i])
        Rot_x[1, 2] = -sin(self.alpha[i])
        Rot_x[2, 1] = sin(self.alpha[i])

        A_i = Rot_z * Trans_z * Trans_x * Rot_x

        return A_i

    # Forward Kinematics

    def get_tcp(self, theta, i_unit='r', o_unit='n'):
        """Solve the tcp based on a list of joint values.

        Args:
            theta: A list of joint values. (unit: radian)
            i_unit: Output format. 'r' for radian; 'd' for degree.
            o_unit: Output format. 'n' for np.array; 'p' for ROS Pose.

        Returns:
            The tcp of end-effector joint w.r.t. base joint
        """

        T_06_PT = torch.eye(4).to(self.device)

        if i_unit == 'd':
            theta = [radians(i) for i in theta]

        for i in range(6):
            T_06_PT *= self.tcp_internal(i, theta)

            return T_06_PT
