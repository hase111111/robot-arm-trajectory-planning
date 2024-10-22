
#-*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php

# [説明]
# 軌道計画を行うプログラム
# 1次，3次，5次の多項式を用いて軌道計画を行う

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List

from two_link_robot import TwoLinkRobot
from util import clamp_angle

class TrajectoryPlannerLiner:
    def __init__(self, robot: TwoLinkRobot = TwoLinkRobot()) -> None:
        self._robot: TwoLinkRobot = robot
        self._state: np.ndarray = np.array([0, 0, np.pi, 0])
        self._total_time: float = 1

    def set_time(self, total_time: float) -> None:
        self._total_time: float = float(total_time)

    def set_state(self, start_theta1: float, start_theta2: float, goal_theta1: float, goal_theta2: float) -> None:
        self._state = np.array([clamp_angle(start_theta1), clamp_angle(start_theta2),
                                clamp_angle(goal_theta1), clamp_angle(goal_theta2)])

    def calc(self, time: float) -> np.ndarray:
        if time < 0:
            return np.array([self._state[0], self._state[1]])
        if time > self._total_time:
            return np.array([self._state[2], self._state[3]])

        theta1 = self._state[0] + (self._state[2] - self._state[0]) * time / self._total_time
        theta2 = self._state[1] + (self._state[3] - self._state[1]) * time / self._total_time
        return np.array([theta1, theta2])
    
class TrajectoryPlannerCubic:
    def __init__(self, robot: TwoLinkRobot = TwoLinkRobot()) -> None:
        self._robot: TwoLinkRobot = robot
        self._state: np.ndarray = np.array([0, 0, np.pi, 0])
        self._delta_state: np.ndarray = np.array([0, 0, 0, 0])
        self._total_time: float = 1

    def set_time(self, total_time: float) -> None:
        self._total_time: float = float(total_time)

    def set_state(self, start_theta1: float, start_theta2: float, goal_theta1: float, goal_theta2: float) -> None:
        self._state = np.array([clamp_angle(start_theta1), clamp_angle(start_theta2),
                                clamp_angle(goal_theta1), clamp_angle(goal_theta2)])

    def set_velocity(self, start_dtheta1: float, start_dtheta2: float, goal_dtheta1: float, goal_dtheta2: float) -> None:
        self._delta_state = np.array([start_dtheta1, start_dtheta2, goal_dtheta1, goal_dtheta2])

    def calc(self, time: float) -> np.ndarray:
        if time < 0:
            return np.array([self._state[0], self._state[1]])
        if time > self._total_time:
            return np.array([self._state[2], self._state[3]])
        
        a0 = [self._state[0], self._state[1]]
        a1 = [self._delta_state[0], self._delta_state[1]]
        a2 = [3 / self._total_time**2 * (self._state[2] - self._state[0]) - 2 / self._total_time * self._delta_state[0],
              3 / self._total_time**2 * (self._state[3] - self._state[1]) - 2 / self._total_time * self._delta_state[1]]
        a3 = [-2 / self._total_time**3 * (self._state[2] - self._state[0]) + 1 / self._total_time**2 * self._delta_state[0],    
              -2 / self._total_time**3 * (self._state[3] - self._state[1]) + 1 / self._total_time**2 * self._delta_state[1]]

        theta1 = a0[0] + a1[0] * time + a2[0] * time**2 + a3[0] * time**3
        theta2 = a0[1] + a1[1] * time + a2[1] * time**2 + a3[1] * time**3
        return np.array([theta1, theta2])

class TrajectoryPlannerQuintic:
    def __init__(self, robot: TwoLinkRobot = TwoLinkRobot()) -> None:
        self._robot: TwoLinkRobot = robot
        self._state: np.ndarray = np.array([0, 0, np.pi, 0])
        self._delta_state: np.ndarray = np.array([0, 0, 0, 0])
        self._delta2_state: np.ndarray = np.array([0, 0, 0, 0])
        self._total_time: float = 1

    def set_time(self, total_time: float) -> None:
        self._total_time: float = float(total_time)

    def set_state(self, start_theta1: float, start_theta2: float, goal_theta1: float, goal_theta2: float) -> None:
        self._state = np.array([clamp_angle(start_theta1), clamp_angle(start_theta2),
                                clamp_angle(goal_theta1), clamp_angle(goal_theta2)])

    def set_velocity(self, start_dtheta1: float, start_dtheta2: float, goal_dtheta1: float, goal_dtheta2: float) -> None:
        self._delta_state = np.array([start_dtheta1, start_dtheta2, goal_dtheta1, goal_dtheta2])

    def set_acceleration(self, start_ddtheta1: float, start_ddtheta2: float, goal_ddtheta1: float, goal_ddtheta2: float) -> None:
        self._delta2_state = np.array([start_ddtheta1, start_ddtheta2, goal_ddtheta1, goal_ddtheta2])

    def calc(self, time: float) -> np.ndarray:
        if time < 0:
            return np.array([self._state[0], self._state[1]])
        if time > self._total_time:
            return np.array([self._state[2], self._state[3]])
        
        a0 = [self._state[0], self._state[1]]
        a1 = [self._delta_state[0], self._delta_state[1]]
        a2 = [self._delta2_state[0] / 2, self._delta2_state[1] / 2]
        a3 = [10 / self._total_time**3 * (self._state[2] - self._state[0]) - 6 / self._total_time**2 * self._delta_state[0] - 3 / self._total_time * self._delta2_state[0],
              10 / self._total_time**3 * (self._state[3] - self._state[1]) - 6 / self._total_time**2 * self._delta_state[1] - 3 / self._total_time * self._delta2_state[1]]
        a4 = [-15 / self._total_time**4 * (self._state[2] - self._state[0]) + 8 / self._total_time**3 * self._delta_state[0] + 7 / self._total_time**2 * self._delta2_state[0],
              -15 / self._total_time**4 * (self._state[3] - self._state[1]) + 8 / self._total_time**3 * self._delta_state[1] + 7 / self._total_time**2 * self._delta2_state[1]]
        a5 = [6 / self._total_time**5 * (self._state[2] - self._state[0]) - 3 / self._total_time**4 * self._delta_state[0] - 3 / self._total_time**3 * self._delta2_state[0],
              6 / self._total_time**5 * (self._state[3] - self._state[1]) - 3 / self._total_time**4 * self._delta_state[1] - 3 / self._total_time**3 * self._delta2_state[1]]

        theta1 = a0[0] + a1[0] * time + a2[0] * time**2 + a3[0] * time**3 + a4[0] * time**4 + a5[0] * time**5
        theta2 = a0[1] + a1[1] * time + a2[1] * time**2 + a3[1] * time**3 + a4[1] * time**4 + a5[1] * time**5
        return np.array([theta1, theta2])

def main():
    robot = TwoLinkRobot()
    planner = TrajectoryPlannerQuintic(robot)
    start_theta1 = 0
    start_theta2 = 0
    goal_theta1 = np.pi / 2
    goal_theta2 = np.pi / 2 * 3

    total_time = 1
    time_division = 100

    planner.set_time(total_time)
    planner.set_state(start_theta1, start_theta2, goal_theta1, goal_theta2)

    time = np.linspace(0, total_time, time_division)
    trajectory = np.array([planner.calc(t) for t in time])

    # Animation
    fig, ax = plt.subplots()

    def update(frame):
        ax.cla()
        ax.set_aspect('equal')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        robot.theta1 = trajectory[frame][0]
        robot.theta2 = trajectory[frame][1]
        robot.plot(ax)

    ani = animation.FuncAnimation(fig, update, frames=len(time), interval=total_time * 1000 / len(time))
    plt.show()

    # t - theta1, theta2   t - dtheta1, dtheta2  t - ddtheta1, ddtheta2
    fig2, ax2 = plt.subplots(3, 2)
    ax2[0, 0].plot(time, trajectory[:, 0])
    ax2[0, 0].set_title("theta1")
    ax2[0, 1].plot(time, trajectory[:, 1])
    ax2[0, 1].set_title("theta2")
    ax2[1, 0].plot(time, np.gradient(trajectory[:, 0], time))
    ax2[1, 0].set_title("dtheta1")
    ax2[1, 1].plot(time, np.gradient(trajectory[:, 1], time))
    ax2[1, 1].set_title("dtheta2")
    ax2[2, 0].plot(time, np.gradient(np.gradient(trajectory[:, 0], time), time))
    ax2[2, 0].set_title("ddtheta1")
    ax2[2, 1].plot(time, np.gradient(np.gradient(trajectory[:, 1], time), time))
    ax2[2, 1].set_title("ddtheta2")
    plt.show()


if __name__ == '__main__':
    main()
