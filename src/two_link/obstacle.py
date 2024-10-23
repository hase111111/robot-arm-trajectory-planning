
#-*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php

# [説明]
# ロボットと衝突する障害物を定義するプログラム

import abc
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import axes

from two_link_robot import TwoLinkRobot

class Obstacle(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def is_collision(self, robot: TwoLinkRobot) -> bool:
        raise NotImplementedError()

class CircleObstacle(Obstacle):
    def __init__(self, x: float, y: float, r: float) -> None:
        self.__x: float = x
        self.__y: float = y
        self.__r: float = r

    def is_collision(self, robot: TwoLinkRobot) -> bool:
        x1, y1, x2, y2 = robot.forward_kinematics()
        ox, oy = robot.param.origin
        if (x1 - self.__x)**2 + (y1 - self.__y)**2 < self.__r**2:
            return True
        if (x2 - self.__x)**2 + (y2 - self.__y)**2 < self.__r**2:
            return True
        if (ox - self.__x)**2 + (oy - self.__y)**2 < self.__r**2:
            return True
        
        # 中点も含めて判定
        m1x, m1y = (x1 + x2) / 2, (y1 + y2) / 2
        if (m1x - self.__x)**2 + (m1y - self.__y)**2 < self.__r**2:
            return True
        
        m2x, m2y = (ox + x1) / 2, (oy + y1) / 2
        if (m2x - self.__x)**2 + (m2y - self.__y)**2 < self.__r**2:
            return True

        return False
    
    def plot(self, ax: axes.Axes) -> None:
        circle = Circle(xy=(self.__x, self.__y), radius=self.__r, fill=True, color='black')
        ax.add_patch(circle)

def main():
    robot = TwoLinkRobot()
    robot.theta1 = 0
    robot.theta2 = 0

    obstacle = CircleObstacle(0.5, 0.5, 0.3)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    robot.plot(ax)
    obstacle.plot(ax)

    plt.show()

if __name__ == '__main__':
    main()
