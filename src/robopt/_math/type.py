"""provides type aliases and functions for type checking"""

# -*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php

try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .._util.type_check import _type_checked


TransMatrix: TypeAlias = NDArray[np.float64]
RotationMatrix: TypeAlias = NDArray[np.float64]
PositionVector: TypeAlias = NDArray[np.float64]


def is_trans_matrix(matrix: NDArray) -> bool:
    """
    Check if the input matrix is a transformation matrix.

    Parameters
    ----------
    matrix : NDArray
        Input matrix.

    Returns
    -------
    bool
        True if the input matrix is a transformation matrix.
    """
    if matrix.shape == (4, 4):
        return True
    else:
        return False


def make_identity_trans_matrix() -> TransMatrix:
    """
    Make an identity transformation matrix.

    Returns
    -------
    TransMatrix
        Identity transformation matrix.
    """
    return np.eye(4)


def is_rot_matrix(matrix: NDArray) -> bool:
    """
    Check if the input matrix is a rotation matrix.

    Parameters
    ----------
    matrix : NDArray
        Input matrix.

    Returns
    -------
    bool
        True if the input matrix is a rotation matrix.
    """

    if matrix.shape == (3, 3):
        return True
    else:
        return False


def make_identity_rot_matrix() -> RotationMatrix:
    """
    Make an identity rotation matrix.

    Returns
    -------
    RotationMatrix
        Identity rotation matrix.
    """
    return np.eye(3)


def is_pos_vector(vector: PositionVector) -> bool:
    """
    Check if the input vector is a position vector(1 x 3).

    Parameters
    ----------
    vector : PositionVector
        Input vector.

    Returns
    -------
    bool
        True if the input vector is a position vector (1 x 3).
    """

    if vector.shape == (3,):
        return True
    else:
        return False


def make_pos_vector(x: float, y: float, z: float) -> PositionVector:
    """
    Make a position vector from x, y, and z.

    Parameters
    ----------
    x : float
        x coordinate.
    y : float
        y coordinate.
    z : float
        z coordinate.

    Returns
    -------
    PositionVector
        Position vector.
    """
    x = _type_checked(x, float)
    y = _type_checked(y, float)
    z = _type_checked(z, float)

    return np.array([x, y, z])


def make_zero_pos_vector() -> PositionVector:
    """
    Make a zero position vector.

    Returns
    -------
    PositionVector
        Zero position vector.
    """
    return np.zeros(3)
