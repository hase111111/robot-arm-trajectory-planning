
#-*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php

# [説明]
# 2リンクロボットのパラメータを入力するための構造体と
# 2リンクロボットの色のパラメータを入力するための構造体

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

    def __str__(self) -> str:
        l_str:str = f'link1: {self.link1}, link2: {self.link2} '
        o_str:str = f'origin: {self.origin}\n'
        # 角度は rad -> deg に変換して表示
        r:float = 180 / np.pi
        theta1_str:str = f'theta1_min: {self.theta1_min * r}, theta1_max: {self.theta1_max * r}\n'
        theta2_str:str = f'theta2_min: {self.theta2_min * r}, theta2_max: {self.theta2_max * r}'
        return l_str + o_str + theta1_str + theta2_str

class TwoLinkRobotColorParam:
    def __init__(self):
        self.link1_color = 'blue'
        self.link2_color = 'blue'
        self.origin_color = 'black'
        self.joint1_color = 'black'
        self.joint2_color = 'black'
        self.link_width = 5
        self.joint_size = 0.05
