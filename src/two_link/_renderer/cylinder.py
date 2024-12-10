# -*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php


import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # type: ignore


import gravibot._math as _math


def draw_cylinder3d(
    ax: Axes3D,
    radius: float,
    height: float,
    pos: _math.PositionVector = _math.make_zero_pos_vector(),
    rot: _math.RotationMatrix = _math.make_identity_rot_matrix(),
    num_slices: int = 20,
    num_stacks: int = 2,
    color: str = "blue",
) -> None:
    """
    指定した座標と向きで円柱を描画する関数
    デフォルトではz軸方向に高さが伸びる円柱を描画する
    """

    if num_slices < 3:
        raise ValueError("num_slices must be 3 or more")

    num_slices += 1  # 端点を含めるため+1

    # θとzの値を生成
    theta = np.linspace(0.0, 2.0 * np.pi, num_slices)
    z = np.linspace(0.0, height, num_stacks)

    # メッシュグリッドを作成
    theta, z = np.meshgrid(theta, z)

    # 円柱の元の座標を計算（円柱側面）
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # 上面と下面の元の座標を計算
    theta_cap = np.linspace(0.0, 2.0 * np.pi, num_slices)
    x_cap = radius * np.cos(theta_cap)
    y_cap = radius * np.sin(theta_cap)
    z_top = np.full_like(theta_cap, height)  # 上面のz座標
    z_bottom = np.zeros_like(theta_cap)  # 下面のz座標

    # 平行移動して円柱の中心を原点に合わせる
    center_translation = _math.get_trans4x4(0.0, 0.0, -height / 2.0)

    # 全体の変換行列を適用
    full_transform = _math.make_trans_by_pos_rot(rot, pos) @ center_translation

    # 側面座標に変換を適用
    points_side = np.vstack((x.ravel(), y.ravel(), z.ravel(), np.ones_like(x.ravel())))
    transformed_side = np.dot(full_transform, points_side)
    x, y, z = (
        transformed_side[0].reshape(x.shape),
        transformed_side[1].reshape(y.shape),
        transformed_side[2].reshape(z.shape),
    )

    # 上面座標に変換を適用
    points_top = np.vstack((x_cap, y_cap, z_top, np.ones_like(x_cap)))
    transformed_top = np.dot(full_transform, points_top)
    x_cap_top, y_cap_top, z_top = (
        transformed_top[0],
        transformed_top[1],
        transformed_top[2],
    )
    epsilon = 1e-6
    x_cap_top += np.random.uniform(-epsilon, epsilon, size=x_cap.shape)
    y_cap_top += np.random.uniform(-epsilon, epsilon, size=y_cap.shape)

    # 下面座標に変換を適用
    points_bottom = np.vstack((x_cap, y_cap, z_bottom, np.ones_like(x_cap)))
    transformed_bottom = np.dot(full_transform, points_bottom)
    x_cap_bottom, y_cap_bottom, z_bottom = (
        transformed_bottom[0],
        transformed_bottom[1],
        transformed_bottom[2],
    )
    x_cap_bottom += np.random.uniform(-epsilon, epsilon, size=x_cap.shape)
    y_cap_bottom += np.random.uniform(-epsilon, epsilon, size=y_cap.shape)

    # 円柱の表面をプロット
    ax.plot_surface(
        x, y, z, rstride=1, cstride=1, alpha=1.0, color=color, edgecolor="none"
    )

    # 上面と下面をプロット（同じ色）
    ax.plot_trisurf(
        x_cap_top, y_cap_top, z_top, color=color, alpha=1.0, edgecolor="none"
    )  # 上面
    ax.plot_trisurf(
        x_cap_bottom, y_cap_bottom, z_bottom, color=color, alpha=1.0, edgecolor="none"
    )  # 下面


def draw_cylinder3d_by_trans(
    ax: Axes3D,
    radius: float,
    height: float,
    trans: _math.TransMatrix = _math.make_identity_trans_matrix(),
    num_slices: int = 20,
    num_stacks: int = 2,
    color: str = "blue",
) -> None:
    """
    指定した変換行列で円柱を描画する関数
    デフォルトではz軸方向に高さが伸びる円柱を描画する
    """

    pos = _math.conv_trans2pos(trans)
    rot = _math.conv_trans2rot(trans)

    draw_cylinder3d(ax, radius, height, pos, rot, num_slices, num_stacks, color)
