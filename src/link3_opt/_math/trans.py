"""provide functions to generate transformation matrices"""

# -*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php


import numpy as np

from .axis import _axis_name_check
from .type import TransMatrix, is_trans_matrix
from .._util.type_check import _type_checked


def get_rot4x4(axis: str, theta: float) -> TransMatrix:
    """
    指定された1軸周りの同時変換行列を生成する関数．
    同時変換行列は4x4の行列である

    Parameters
    ----------
    axis : str
        回転軸．'x', 'y', 'z'のいずれか．
    theta : float
        回転角．単位はラジアン．

    Returns
    -------
    rot_mat : TransMatrix
        4x4の同時変換行列．
    """
    a = _axis_name_check(axis)
    theta = _type_checked(theta, float)

    # 回転行列（同時変換行列）の生成
    if a == "x":
        rot_mat = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(theta), -np.sin(theta), 0],
                [0, np.sin(theta), np.cos(theta), 0],
                [0, 0, 0, 1],
            ]
        )
    elif a == "y":
        rot_mat = np.array(
            [
                [np.cos(theta), 0, np.sin(theta), 0],
                [0, 1, 0, 0],
                [-np.sin(theta), 0, np.cos(theta), 0],
                [0, 0, 0, 1],
            ]
        )
    else:
        rot_mat = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0, 0],
                [np.sin(theta), np.cos(theta), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

    return rot_mat


def get_trans4x4(x: float, y: float, z: float) -> TransMatrix:
    """
    指定された方向の移動する同時変換行列を生成する関数．

    Parameters
    ----------
    x : float
        x軸方向の移動量 [m].
    y : float
        y軸方向の移動量 [m].
    z : float
        z軸方向の移動量 [m].

    Returns
    -------
    trans_mat : TransMatrix
        4x4の同時変換行列．
    """

    x = _type_checked(x, float)  # x座標
    y = _type_checked(y, float)  # y座標
    z = _type_checked(z, float)  # z座標

    # 移動行列
    return np.array(
        [
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def zero_small_values4x4(trans: TransMatrix) -> TransMatrix:
    """
    小さな値を0に置き換える関数。

    Parameters
    ----------
    trans : TransMatrix
        4x4の同次変換行列。

    Returns
    -------
    TransMatrix
        修正された同次変換行列。
    """
    trans = trans.copy()  # 元の行列を保持するためにコピーを作成

    if is_trans_matrix(trans) is False:
        # 入力が4x4行列でなければ例外を投げる
        raise ValueError("Input matrix must be 4x4.")

    eps: float = 1e-10
    trans[np.abs(trans) <= eps] = 0.0  # 小さな値をゼロに置き換え
    return trans
