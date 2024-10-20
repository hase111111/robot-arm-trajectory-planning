
import numpy as np

from util import clamp_angle

class TwoLinkRobotParam:
    def __init__(self):
        self.link1 = 1.0
        self.link2 = 1.0
        self.origin = np.array([0, 0])
        self.theta1_min = -np.pi
        self.theta1_max = np.pi
        self.theta2_min = -np.pi
        self.theta2_max = np.pi

        @property
        def theta1_min(self):
            return self._theta1_min
        
        @theta1_min.setter
        def theta1_min(self, value):
            self._theta1_min = clamp_angle(value)

        @property
        def theta1_max(self):
            return self._theta1_max
        
        @theta1_max.setter
        def theta1_max(self, value):
            self._theta1_max = clamp_angle(value)

        @property
        def theta2_min(self):
            return self._theta2_min
        
        @theta2_min.setter
        def theta2_min(self, value):
            self._theta2_min = clamp_angle(value)

        @property
        def theta2_max(self):
            return self._theta2_max
        
        @theta2_max.setter
        def theta2_max(self, value):
            self._theta2_max = clamp_angle(value)

class TwoLinkRobotColorParam:
    def __init__(self):
        self.link1_color = 'blue'
        self.link2_color = 'blue'
        self.origin_color = 'black'
        self.joint1_color = 'black'
        self.joint2_color = 'black'
        self.link_width = 5
        self.joint_size = 0.05
