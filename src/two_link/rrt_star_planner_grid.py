
#-*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List
from typing import Tuple

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
        self.grid_num = 100

    def set_time(self, total_time: float) -> None:
        self._total_time: float = float(total_time)
        self.already_done = False

    def set_state(self, start_theta1: float, start_theta2: float, goal_theta1: float, goal_theta2: float) -> None:
        self._state = np.array([clamp_angle(start_theta1), clamp_angle(start_theta2),
                                clamp_angle(goal_theta1), clamp_angle(goal_theta2)])
        self.already_done = False

    def calc(self, time: float) -> np.ndarray:
        if not self.already_done:
            if not self.__rrt_star():
                return np.array([self._state[0], self._state[1]])

        if time < 0:
            return np.array([self._state[0], self._state[1]])
        if time > self._total_time:
            return np.array([self._state[2], self._state[3]])
        
        index = int(time / self._total_time * (len(self.result[0]) - 1))
        return np.array([self.result[0][index], self.result[1][index]])
    
    def __rrt_star(self) -> bool:
        print('RRT*')
        # 最短20stepぐらいでゴールに到達するようにする
        step = math.sqrt((self._state[2] - self._state[0])**2 + (self._state[3] - self._state[1])**2) / 20
        print('step: ' + str(step))

        grid = self.__make_collision_grid()

        # ノードのリスト
        nodes: List[TrajectoryPlannerLiner.Node] = []
        nodes.append(TrajectoryPlannerLiner.Node(self._state[0], self._state[1], -1))

        cnt = 0
        max_cnt = 10000
        success = False

        # [-π ~ π, -π ~ π]の範囲でランダムな点を生成
        # nodesの中から最も近いノードを探し、その方向にstepだけ進んだ点を新たなノードとしてchildに追加
        # 追加されたノードとゴールの距離がstep以下なら終了
        # max_cnt回繰り返しても終了しなかったら終了

        while cnt < max_cnt:
            cnt += 1
            x = np.random.uniform(-np.pi, np.pi)
            y = np.random.uniform(-np.pi, np.pi)

            min_distance2 = float('inf')
            min_index = -1
            for i, node in enumerate(nodes):
                distance2 = (node.x - x)**2 + (node.y - y)**2
                if distance2 < min_distance2:
                    min_distance2 = distance2
                    min_index = i

            theta1 = nodes[min_index].x
            theta2 = nodes[min_index].y
            theta1 += step * math.cos(math.atan2(y - theta2, x - theta1))
            theta2 += step * math.sin(math.atan2(y - theta2, x - theta1))
            if grid[self.__get_index(theta1, theta2)]:
                continue

            # ここまで来たら，x_new(theta1, theta2)を追加
            # ただし，root からの経路が最も近くなるようなノードを探して，そのノードに接続する

            min_distance = float('inf')
            min_index = -1
            for i, node in enumerate(nodes):
                distance = math.sqrt((node.x - theta1)**2 + (node.y - theta2)**2)
                if distance < min_distance:
                    min_distance = distance
                    min_index = i

            nodes.append(TrajectoryPlannerLiner.Node(theta1, theta2, min_index))
            nodes[min_index].children.append(len(nodes) - 1)

            if (theta1 - self._state[2])**2 + (theta2 - self._state[3])**2 < step**2:
                success = True
                break

        if not success:
            print('Fail')
            self.already_done = True
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
    
    def __make_collision_grid(self) -> np.ndarray:
        print('now making collision grid')
        # [-π ~ π,-π ~ π] を self.grid_num x self.grid_num のグリッドに分割
        # 障害物がある場所はTrue, ない場所はFalse
        self.collision_grid = np.zeros((self.grid_num, self.grid_num), dtype=bool)
        for i in range(self.grid_num):
            for j in range(self.grid_num):
                ang1, ang2 = self.__get_angle(i, j)
                self._robot.theta1 = ang1
                self._robot.theta2 = ang2
                if self._obstacle.is_collision(self._robot):
                    self.collision_grid[i, j] = True

        print('finish making collision grid')
        return self.collision_grid

    def __get_index(self, ang1: float, ang2: float) -> Tuple[int, int]:
        # 角度をグリッドのインデックスに変換
        # 0 ~ 2π を 0 ~ self.grid_num に変換
        return int(ang1 / (2 * np.pi) * self.grid_num), int(ang2 / (2 * np.pi) * self.grid_num)
    
    def __get_angle(self, i: int, j: int) -> Tuple[float, float]:
        # グリッドのインデックスを角度に変換
        return i / self.grid_num * 2 * np.pi, j / self.grid_num * 2 * np.pi
    
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
    planner.set_state(0, 0, np.pi / 2, np.pi / 2)

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
