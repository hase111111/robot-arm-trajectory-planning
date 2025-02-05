"""provide functions for converting one matrix to another"""

# -*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php


import numpy as np

from .type import TransMatrix, RotationMatrix, PositionVector
from .type import is_pos_vector, is_rot_matrix, is_trans_matrix


def conv_trans2pos(trans: TransMatrix) -> PositionVector:
    """
    同時変換行列から位置ベクトルを抽出する．

    Parameters
    ----------
    trans : TransMatrix
        4x4の同時変換行列．

    Returns
    -------
    pos : PositionVector
        1x3の位置ベクトル．
    """

    # 入力が4x4行列であることを確認
    if is_trans_matrix(trans) is False:
        raise ValueError("Input matrix must be 4x4.")

    return np.array([trans[0][3], trans[1][3], trans[2][3]]).transpose()


def conv_trans2rot(trans: TransMatrix) -> RotationMatrix:
    """
    同時変換行列から回転行列を抽出する．

    Parameters
    ----------
    trans : TransMatrix
        4x4の同次変換行列．

    Returns
    -------
    rot : RotationMatrix
        3x3の回転行列．
    """

    # 入力が4x4行列であることを確認
    if is_trans_matrix(trans) is False:
        raise ValueError("Input matrix must be 4x4.")

    # 同次変換行列の上3x3部分を回転行列として抽出
    rot = trans[:3, :3]

    return rot


def make_trans_by_pos_rot(rot: RotationMatrix, pos: PositionVector) -> TransMatrix:
    """
    位置ベクトルと回転行列から同次変換行列を構築する．

    Parameters
    ----------
    rot : RotationMatrix
        3x3の回転行列．
    pos : PositionVector
        1x3または3x1の平行移動ベクトル．

    Returns
    -------
    trans : TransMatrix
        4x4の同次変換行列．
    """
    # 入力が適切なサイズであることを確認
    if is_rot_matrix(rot) is False or is_pos_vector(pos) is False:
        raise ValueError("Input matrix must be 3x3 and 1x3 or 3x1.")

    # 平行移動ベクトルを1x3に整形
    pos = np.ravel(pos)

    # 4x4の同次変換行列を構築
    trans = np.eye(4)
    trans[:3, :3] = rot
    trans[:3, 3] = pos

    return trans
