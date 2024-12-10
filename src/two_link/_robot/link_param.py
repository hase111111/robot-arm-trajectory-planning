"""provide LinkParam class"""

# -*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php


import numpy as np

from .._math.type import TransMatrix
from .._math.trans import get_rot4x4, get_trans4x4, zero_small_values4x4
from .._util.type_check import _type_checked


class LinkParam:
    """class for link parameters"""

    def __init__(
        self,
        a: float,
        alpha: float,
        d: float,
        theta: float,
        *,
        is_rot_axis: bool = True,
        min_val: float = -np.pi,
        max_val: float = np.pi,
    ):
        self._a = _type_checked(a, float)
        self._alpha = _type_checked(alpha, float)
        self._d = _type_checked(d, float)
        self._theta = _type_checked(theta, float)
        self.is_rot_axis = _type_checked(is_rot_axis, bool)
        self.min_val = _type_checked(min_val, float)
        self.max_val = _type_checked(max_val, float)

    def get_trans_mat(self) -> TransMatrix:
        """return link's A matrix"""
        ans = (
            get_rot4x4("z", self.theta)
            @ get_trans4x4(0.0, 0.0, self.d)
            @ get_trans4x4(self.a, 0.0, 0.0)
            @ get_rot4x4("x", self.alpha)
        )
        return zero_small_values4x4(ans)

    def set_val(self, val: float) -> None:
        """set d and theta"""

        val = _type_checked(val, float)

        if not self.min_val <= val <= self.max_val:
            raise ValueError(
                f"theta should be in range [{self.min_val}, {self.max_val}]"
            )
        if self.is_rot_axis:
            self._theta = val
        else:
            self._d = val

    @property
    def a(self) -> float:
        """getter for a, a is read-only"""
        return self._a

    @a.setter
    def a(self, _):
        raise AttributeError("a is read-only")

    @property
    def alpha(self) -> float:
        """getter for alpha, alpha is read-only"""
        return self._alpha

    @alpha.setter
    def alpha(self, _):
        raise AttributeError("alpha is read-only")

    @property
    def d(self) -> float:
        """getter for d, d can being changed by set method"""
        return self._d

    @d.setter
    def d(self, _):
        raise AttributeError("Should use set_val method")

    @property
    def theta(self) -> float:
        """getter for theta, theta can being changed by set method"""
        return self._theta

    @theta.setter
    def theta(self, _):
        raise AttributeError("Should use set_val method")
