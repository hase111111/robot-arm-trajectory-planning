
#-*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List

from two_link_robot import TwoLinkRobot
from two_link_robot_param import TwoLinkRobotParam
from obstacle import Obstacle, CircleObstacle
from util import clamp_angle

class TrajectoryPlannerLiner:
    class Node:
        def __init__(self, x: float, y: float, parent: int) -> None:
            self.x: float = x
            self.y: float = y
            self.parent: int = parent
            self.children: List[int] = []

    def __init__(self, robot: TwoLinkRobot = TwoLinkRobot(), obstacle: Obstacle = CircleObstacle(1, 1, 0.5)) -> None:
        self._robot: TwoLinkRobot = robot
        self._obstacle: Obstacle = obstacle
        self._state: np.ndarray = np.array([0, 0, np.pi, 0])
        self._total_time: float = 1
        self.already_done: bool = False
        self.result: List[np.ndarray] = []

    def set_time(self, total_time: float) -> None:
        self._total_time: float = float(total_time)
        self.already_done = False

    def set_state(self, start_theta1: float, start_theta2: float, goal_theta1: float, goal_theta2: float) -> None:
        self._state = np.array([clamp_angle(start_theta1), clamp_angle(start_theta2),
                                clamp_angle(goal_theta1), clamp_angle(goal_theta2)])
        self.already_done = False

    def calc(self, time: float) -> np.ndarray:
        if not self.already_done:
            self.__rrt_star()

        if time < 0:
            return np.array([self._state[0], self._state[1]])
        if time > self._total_time:
            return np.array([self._state[2], self._state[3]])
        
        index = int(time / self._total_time * (len(self.result[0]) - 1))
        return np.array([self.result[0][index], self.result[1][index]])
    
    def __rrt_star(self) -> bool:
        print('RRT*')
        step = 0.05

        # ノードのリスト
        nodes: List[TrajectoryPlannerLiner.Node] = []
        nodes.append(TrajectoryPlannerLiner.Node(self._state[0], self._state[1], -1))

        cnt = 0
        max_cnt = 100000
        success = False

        # [-π ~ π, -π ~ π]の範囲でランダムな点を生成
        # この点が障害物に含まれていたら生成し直す
        # nodesの中から最も近いノードを探し、その方向にstepだけ進んだ点を新たなノードとしてchildに追加
        # 追加されたノードとゴールの距離がstep以下なら終了
        # max_cnt回繰り返しても終了しなかったら終了

        while cnt < max_cnt:
            cnt += 1
            x = np.random.uniform(-np.pi, np.pi)
            y = np.random.uniform(-np.pi, np.pi)

            min_distance = float('inf')
            min_index = -1
            for i, node in enumerate(nodes):
                distance = math.sqrt((node.x - x)**2 + (node.y - y)**2)
                if distance < min_distance:
                    min_distance = distance
                    min_index = i

            theta1 = nodes[min_index].x
            theta2 = nodes[min_index].y
            theta1 += step * math.cos(math.atan2(y - theta2, x - theta1))
            theta2 += step * math.sin(math.atan2(y - theta2, x - theta1))
            self._robot.theta1 = theta1
            self._robot.theta2 = theta2
            if self._obstacle.is_collision(self._robot):
                continue

            nodes.append(TrajectoryPlannerLiner.Node(theta1, theta2, min_index))
            nodes[min_index].children.append(len(nodes) - 1)

            if math.sqrt((theta1 - self._state[2])**2 + (theta2 - self._state[3])**2) < step:
                success = True
                break

        if not success:
            print('Fail')
            return False
        
        # ゴールから逆にたどっていく
        index = len(nodes) - 1
        theta1_result = []
        theta2_result = []
        while index != -1:
            theta1_result.append(nodes[index].x)
            theta2_result.append(nodes[index].y)
            index = nodes[index].parent

        self.result = [theta1_result[::-1], theta2_result[::-1]]
        self.already_done = True
        print('Success')
        print('path: ' + str(len(self.result[0])))
        print('node: ' + str(len(nodes)))
        return True
    
def main():
    # ロボットのパラメータ
    robot_param = TwoLinkRobotParam()
    robot_param.link1 = 1.0
    robot_param.link2 = 1.0
    robot_param.origin = np.array([0, 0])
    robot_param.theta1_min = -np.pi
    robot_param.theta1_max = np.pi
    robot_param.theta2_min = -np.pi
    robot_param.theta2_max = np.pi

    total_time = 1
    time_division = 100

    # ロボットのクラスを作成
    robot = TwoLinkRobot(robot_param)

    # 障害物のパラメータ
    obstacle = CircleObstacle(1.5, 1.5, 0.5)

    # 軌道計画
    planner = TrajectoryPlannerLiner(robot, obstacle)
    planner.set_time(1)
    planner.set_state(0, 0, np.pi / 2, 0)

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
        obstacle.plot(ax)

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
