
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

class RRTStarPlannerGrid:
    class Node:
        def __init__(self, x: float, y: float, parent: int, cost:float = 0) -> None:
            self.x: float = x
            self.y: float = y
            self.parent: int = parent
            self.cost: float = cost

    def __init__(self, robot: TwoLinkRobot = TwoLinkRobot(), 
                 obstacle: Obstacle = CircleObstacle(1, 1, 0.5),
                 *, animation = False) -> None:
        self._robot: TwoLinkRobot = robot
        self._obstacle: Obstacle = obstacle
        self._state: np.ndarray = np.array([0, 0, np.pi, 0])
        self._total_time: float = 1
        self.already_done: bool = False
        self.result: List[np.ndarray] = []
        self.grid_num = 100
        self.animation = animation

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
            else:
                # 10 回ポストプロセスを行う
                for _ in range(10):
                    self.__post_processing()

        if time < 0:
            return np.array([self._state[0], self._state[1]])
        if time > self._total_time:
            return np.array([self._state[2], self._state[3]])
        
        index = int(time / self._total_time * (len(self.result[0]) - 1))
        return np.array([self.result[0][index], self.result[1][index]])
    
    def __rrt_star(self) -> bool:
        print('Now Start RRT')
        # 最短10stepぐらいでゴールに到達するようにする
        step = math.sqrt((self._state[2] - self._state[0])**2 + (self._state[3] - self._state[1])**2) / 10
        print('step: ' + str(step))

        if self.animation:
            self.__init_plot()

        self.grid = self.__make_collision_grid()

        # ノードのリスト
        nodes: List[RRTStarPlannerGrid.Node] = []
        nodes.append(RRTStarPlannerGrid.Node(self._state[0], self._state[1], -1, 0))

        cnt = 0
        max_cnt = 10000
        success = False

        # [-π ~ π, -π ~ π]の範囲でランダムな点を生成
        # nodesの中から最も近いノードを探し、その方向にstepだけ進んだ点を新たなノードとしてchildに追加
        # 追加されたノードとゴールの距離がstep以下なら終了
        # max_cnt回繰り返しても終了しなかったら終了

        while cnt < max_cnt:
            cnt += 1

            # [-π ~ π, -π ~ π]の範囲でランダムな点を生成
            rand1, rand2 = self.__get_random_angle()

            # nodesの中から最も近いノードを探す
            min_index = self.__get_most_near_index(nodes, rand1, rand2)

            # min_index から x, y に向かってstepだけ進んだ点を新たなノードとして追加
            new_node = self.__streer(rand1, rand2, nodes[min_index], step)

            # 障害物に衝突していたらスキップ
            if self.grid[self.__get_index(new_node.x, new_node.y)]:
                continue

            # 近くにあるノードを探す
            nearinds = self.__find_near_index_list(nodes, new_node, step)

            # 新たなノードの親を決める
            new_node = self.__choose_parent(nodes, new_node, nearinds)

            # 新たなノードを追加
            new_node.parent = min_index
            nodes.append(new_node)

            if (new_node.x - self._state[2])**2 + (new_node.y - self._state[3])**2 < step**2:
                success = True
                break

            if self.animation:
                self.__plot_process(nodes)

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
    
    def __get_random_angle(self) -> Tuple[float, float]:
        return np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)
    
    def __get_most_near_index(self, nodes: List[Node], x: float, y: float) -> int:
        min_distance = float('inf')
        min_index = -1
        for i, node in enumerate(nodes):
            distance = (node.x - x)**2 + (node.y - y)**2
            if distance < min_distance:
                min_distance = distance
                min_index = i
        return min_index
    
    def __streer(self, x: float, y: float, node: Node, step: float) -> Node:
        theta1, theta2 = node.x, node.y
        theta1 += step * math.cos(math.atan2(y - theta2, x - theta1))
        theta2 += step * math.sin(math.atan2(y - theta2, x - theta1))
        # 親を仮に-1にして、ノードを返す
        return RRTStarPlannerGrid.Node(theta1, theta2, -1, node.cost + step)
    
    def __find_near_index_list(self, nodes: List[Node], new_node: Node, step: float) -> List[int]:
        # r = step * 20 * (log(len(nodes)) / len(nodes))^(1/2)
        r = step * 20 * (math.log(len(nodes)) / len(nodes))**(1/2)
        dlist = [(node.x - new_node.x) ** 2 +
                 (node.y - new_node.y) ** 2 for node in nodes]
        nearinds = [dlist.index(i) for i in dlist if i <= r ** 2]
        return nearinds
    
    def __choose_parent(self, nodes: List[Node], new_node: Node, nearinds: List[int]) -> Node:
        if len(nearinds) == 0:
            return new_node

        dlist = []
        for i in nearinds:
            dx = new_node.x - nodes[i].x
            dy = new_node.y - nodes[i].y
            d = math.sqrt(dx ** 2 + dy ** 2)
            theta = math.atan2(dy, dx)
            dlist.append(nodes[i].cost + d)

        mincost = min(dlist)
        minind = nearinds[dlist.index(mincost)]

        if mincost == float("inf"):
            print("mincost is inf")
            return new_node

        new_node.cost = mincost
        new_node.parent = minind

        return new_node

    def rewire(self, nodes: List[Node], new_node: Node, nearinds: List[int]) -> None:
        nnode = len(nodes)
        for i in nearinds:
            nearNode = nodes[i]

            dx = new_node.x - nearNode.x
            dy = new_node.y - nearNode.y
            d = math.sqrt(dx ** 2 + dy ** 2)

            scost = new_node.cost + d

            if nearNode.cost > scost:
                nearNode.parent = nnode - 1
                nearNode.cost = scost

    def __post_processing(self):
        # 2点を選び、それらの間に障害物がないか確認
        # 障害物がない場合、その間にあるノードを削除
        # これを繰り返す
        first_index = 0
        second_index = 2
        while second_index < len(self.result[0]):
            # 中点を取得
            mid_theta1 = (self.result[0][first_index] + self.result[0][second_index]) / 2
            mid_theta2 = (self.result[1][first_index] + self.result[1][second_index]) / 2

            # 中点が障害物に当たるか確認
            if not self.grid[self.__get_index(mid_theta1, mid_theta2)]:
                # 障害物に当たらない場合、first_index と second_index の間にあるノードを中点に変更
                self.result[0][first_index + 1] = mid_theta1
                self.result[1][first_index + 1] = mid_theta2
                first_index += 1
                second_index += 1                
            else:
                first_index += 1
                second_index += 1

        print('finish post processing')

    def __init_plot(self) -> None:
        _, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-np.pi, np.pi)
        self.ax.set_ylim(-np.pi, np.pi)
    
    def __plot_process(self, nodes: List[Node]) -> None:
        # 図をクリアする
        self.ax.cla()
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-np.pi, np.pi)
        self.ax.set_ylim(-np.pi, np.pi)

        # ゴールとスタートを描画
        self.ax.add_patch(plt.Circle((self._state[0], self._state[1]), 0.1, color='blue'))
        self.ax.add_patch(plt.Circle((self._state[2], self._state[3]), 0.1, color='green'))

        # 各ノードについて、親ノードとの間に線を引く
        for i, node in enumerate(nodes):
            if node.parent == -1:
                continue
            self.ax.plot([nodes[node.parent].x, node.x], [nodes[node.parent].y, node.y], color='black', linewidth=0.5)

        # gridの描画
        for i in range(self.grid_num):
            for j in range(self.grid_num):
                if self.grid[i, j]:
                    ang1, ang2 = self.__get_angle(i, j)
                    self.ax.add_patch(plt.Circle((ang1, ang2), 0.01, color='red'))

        plt.draw()
        plt.pause(0.001)     # 更新時間まち
    
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
    obstacle = CircleObstacle(0, 1.6, 0.6)

    # 軌道計画
    planner = RRTStarPlannerGrid(robot, obstacle, animation=False)
    planner.set_time(total_time)
    planner.set_state(0, 0, np.pi, 0)

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
