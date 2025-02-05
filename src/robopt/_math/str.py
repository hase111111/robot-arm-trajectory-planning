"""provide functions to convert math objects to string"""

# -*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php


from .type import PositionVector, RotationMatrix, TransMatrix


def posvec_to_str(posvec: PositionVector) -> str:
    """convert position vector to string"""
    return f"x:{posvec[0]:.3f}, y:{posvec[1]:.3f}, z:{posvec[2]:.3f}"


def rotmat_to_str(rotmat: RotationMatrix) -> str:
    """convert rotation matrix to string"""
    return (
        f"{rotmat[0, 0]:.3f}, {rotmat[0, 1]:.3f}, {rotmat[0, 2]:.3f} \n"
        + f" {rotmat[1, 0]:.3f}, {rotmat[1, 1]:.3f}, {rotmat[1, 2]:.3f} \n "
        + f"{rotmat[2, 0]:.3f}, {rotmat[2, 1]:.3f}, {rotmat[2, 2]:.3f}"
    )


def transmat_to_str(transmat: TransMatrix) -> str:
    """convert transformation matrix to string"""
    return (
        f"{transmat[0, 0]:.3f}, {transmat[0, 1]:.3f}, {transmat[0, 2]:.3f}, {transmat[0, 3]:.3f} \n"
        + f"{transmat[1, 0]:.3f}, {transmat[1, 1]:.3f}, {transmat[1, 2]:.3f}, {transmat[1, 3]:.3f} \n"
        + f"{transmat[2, 0]:.3f}, {transmat[2, 1]:.3f}, {transmat[2, 2]:.3f}, {transmat[2, 3]:.3f} \n"
        + f"{transmat[3, 0]:.3f}, {transmat[3, 1]:.3f}, {transmat[3, 2]:.3f}, {transmat[3, 3]:.3f}"
    )
