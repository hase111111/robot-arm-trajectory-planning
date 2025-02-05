"""provide functions to generate rotation matrix"""

# -*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php


import numpy as np

from .axis import _axis_name_check
from .type import RotationMatrix, is_rot_matrix
from .._util.type_check import _type_checked


def get_rot3x3(axis: str, theta: float) -> RotationMatrix:
    """
    指定された軸周りの回転行列を生成する関数．
    回転行列は3x3の行列である

    Parameters
    ----------
    axis : str
        回転軸．'x', 'y', 'z'のいずれか．
    theta : float
        回転角．単位はラジアン．

    Returns
    -------
    rot_mat : RotationMatrix
        3x3の回転行列．
    """
    a = _axis_name_check(axis)
    theta = _type_checked(theta, float)

    # 回転行列の生成
    if a == "x":
        rot_mat = np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ]
        )
    elif a == "y":
        rot_mat = np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )
    else:
        rot_mat = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )

    return rot_mat


def zero_small_values4x4(trans: RotationMatrix) -> RotationMatrix:
    """
    小さな値を0に置き換える関数。

    Parameters
    ----------
    trans : RotationMatrix
        3x3の同次変換行列。

    Returns
    -------
    RotationMatrix
        修正された同次変換行列。
    """
    trans = trans.copy()  # 元の行列を保持するためにコピーを作成

    if is_rot_matrix(trans) is False:
        # 入力が4x4行列でなければ例外を投げる
        raise ValueError("Input matrix must be 4x4.")

    eps: float = 1e-10
    trans[np.abs(trans) <= eps] = 0.0  # 小さな値をゼロに置き換え
    return trans
