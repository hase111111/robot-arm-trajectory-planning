"""This module is __init__.py of gravibot/_math package."""

# -*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php


from .converter import conv_trans2pos, conv_trans2rot, make_trans_by_pos_rot
from .rot import get_rot3x3
from .str import posvec_to_str, rotmat_to_str, transmat_to_str
from .trans import get_rot4x4, get_trans4x4, zero_small_values4x4
from .type import TransMatrix, RotationMatrix, PositionVector
from .type import (
    is_trans_matrix,
    make_identity_trans_matrix,
    is_rot_matrix,
    make_identity_rot_matrix,
    is_pos_vector,
    make_pos_vector,
    make_zero_pos_vector,
)

__all__ = [
    "get_rot4x4",
    "get_trans4x4",
    "zero_small_values4x4",
    "get_rot3x3",
    "posvec_to_str",
    "rotmat_to_str",
    "transmat_to_str",
    "TransMatrix",
    "RotationMatrix",
    "PositionVector",
    "is_trans_matrix",
    "make_identity_trans_matrix",
    "is_rot_matrix",
    "make_identity_rot_matrix",
    "is_pos_vector",
    "make_pos_vector",
    "make_zero_pos_vector",
    "conv_trans2pos",
    "conv_trans2rot",
    "make_trans_by_pos_rot",
]
