
#-*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php

# [説明]
# matplotlibを使って2リンクロボットの逆運動学を可視化するプログラム
# マウスカーソルの位置にエンドエフェクタを持っていくように2リンクロボットを動かす
# マウスの左クリックで逆運動学解を切り替える

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from two_link_robot_param import TwoLinkRobotParam, TwoLinkRobotColorParam
from two_link_robot import TwoLinkRobot

def main():
    # ロボットのパラメータ
    param = TwoLinkRobotParam()
    param.link1 = 1.0
    param.link2 = 1.0
    param.origin = np.array([0, 0])
    param.theta1_min = -np.pi
    param.theta1_max = np.pi
    param.theta2_min = -np.pi
    param.theta2_max = np.pi

    # ロボットの色のパラメータ
    color_param = TwoLinkRobotColorParam()

    # ロボットのクラスを作成
    robot = TwoLinkRobot(param, color_param)

    # ウィンドウの作成
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    # ロボットの初期姿勢
    theta1 = 0
    theta2 = 0

    # マウスカーソルの位置
    end_effecter_x = 1.5
    end_effecter_y = 1.5

    # 逆運動学解を切り替えるためのフラグ
    other = False

    # ロボットの描画
    x1, y1, x2, y2 = robot.forward_kinematics(theta1, theta2)
    link1, = ax.plot([param.origin[0], x1], [param.origin[1], y1], color=color_param.link1_color)
    link2, = ax.plot([x1, x2], [y1, y2], color=color_param.link2_color)

    # マウスカーソルの位置を表示するための円
    end_effecter_circle = Circle((end_effecter_x, end_effecter_y), 0.1, color='black', fill=False)
    ax.add_patch(end_effecter_circle)

    def on_move(event):
        nonlocal end_effecter_x, end_effecter_y
        end_effecter_x = event.xdata
        end_effecter_y = event.ydata
        if end_effecter_x is None or end_effecter_y is None:
            return
        end_effecter_circle.center = (end_effecter_x, end_effecter_y)
        theta1, theta2 = robot.inverse_kinematics(end_effecter_x, end_effecter_y, other=other)
        if robot.is_in_range(theta1, theta2):
            x1, y1, x2, y2 = robot.forward_kinematics(theta1, theta2)
            link1.set_xdata([param.origin[0], x1])
            link1.set_ydata([param.origin[1], y1])
            link2.set_xdata([x1, x2])
            link2.set_ydata([y1, y2])
            fig.canvas.draw()

    def on_click(event):
        nonlocal theta1, theta2, other
        other = not other
        theta1, theta2 = robot.inverse_kinematics(end_effecter_x, end_effecter_y, other=other)
        x1, y1, x2, y2 = robot.forward_kinematics(theta1, theta2)
        link1.set_xdata([param.origin[0], x1])
        link1.set_ydata([param.origin[1], y1])
        link2.set_xdata([x1, x2])
        link2.set_ydata([y1, y2])
        fig.canvas.draw()

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()

if __name__ == '__main__':
    main()
