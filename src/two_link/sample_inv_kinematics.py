
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

def my_clear(ax):
    ax.clear()
    ax.set_aspect('equal')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

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

    # マウスカーソルの位置
    end_effecter_x = 1.0
    end_effecter_y = 1.0

    # 逆運動学解を切り替えるためのフラグ
    other = False

    # ロボットの描画
    robot.plot(ax)

    def on_move(event):
        nonlocal other, end_effecter_x, end_effecter_y
        x = end_effecter_x if event.xdata is None else float(event.xdata)
        y = end_effecter_y if event.ydata is None else float(event.ydata)
        end_effecter_x = x
        end_effecter_y = y

        my_clear(ax)
        ax.add_patch(Circle([x, y], 0.1, color='black', fill=False))
        robot.inverse_kinematics(x, y, other=other)
        robot.plot(ax)
        fig.canvas.draw()

    def on_click(event):
        nonlocal other, end_effecter_x, end_effecter_y

        # 左クリックのみ受け付ける
        if event.button != 1:
            return
        
        other = not other
        robot.inverse_kinematics(end_effecter_x, end_effecter_y, other=other)

        my_clear(ax)
        ax.add_patch(Circle([end_effecter_x, end_effecter_y], 0.1, color='black', fill=False))
        robot.plot(ax)
        fig.canvas.draw()

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()

if __name__ == '__main__':
    main()
