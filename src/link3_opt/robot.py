"""Robot class for gravibot"""

from mpl_toolkits.mplot3d import Axes3D  # type: ignore
import numpy as np

import ._math as _math
import ._robot as _robot
import ._renderer as _renderer
from ._util.type_check import _type_checked


class Robot:
    """class for robot"""

    def __init__(self, param: _robot.RobotParam):
        if not isinstance(param, _robot.RobotParam):
            raise TypeError(f"param must be RobotParam, not {type(param)}")

        self._param = param
        self._origin = _math.make_zero_pos_vector()

    def get_joint_trans(self, i: int) -> _math.TransMatrix:
        """get the transformation matrix of the i-th joint"""

        i = _type_checked(i, int)
        self._validate_joint_num(i)

        ans = _math.make_identity_trans_matrix()
        for j in range(i + 1):
            ans = ans @ self._param.get_link(j).get_trans_mat()

        return ans

    def get_joint_pos(self, i):
        """get the position of the i-th joint"""

        i = _type_checked(i, int)
        self._validate_joint_num(i)

        return _math.conv_trans2pos(self.get_joint_trans(i))

    def draw(self, ax: Axes3D):
        """draw the robot"""
        for i in range(self._param.get_num_links() - 1):
            _renderer.draw_cylinder3d_by_trans(
                ax,
                1.5,
                3.0,
                self.get_joint_trans(i),
                color="blue",
            )
            self._draw_link(ax, self.get_joint_pos(i), self.get_joint_pos(i + 1))

        self._draw_link(ax, self._origin, self.get_joint_pos(0))
        _renderer.draw_cylinder3d_by_trans(ax, 3, 2.0, color="green")

    def _draw_link(
        self, ax: Axes3D, pos1: _math.PositionVector, pos2: _math.PositionVector
    ) -> None:
        """draw a link between pos1 and pos2"""

        length = float(np.linalg.norm(pos2 - pos1))
        if length < 1e-10:
            return

        pos1to2: _math.PositionVector = (pos2 - pos1) / length
        pos_center: _math.PositionVector = pos1 + (pos2 - pos1) / 2
        z_base = _math.make_pos_vector(0.0, 0.0, 1.0)

        if np.allclose(pos1to2, z_base):
            rotation_matrix = _math.make_identity_rot_matrix()
        else:
            x_new = np.cross(z_base, pos1to2)
            x_new /= np.linalg.norm(x_new)
            y_new = np.cross(pos1to2, x_new)
            rotation_matrix = np.array([x_new, y_new, pos1to2]).T

        _renderer.draw_cylinder3d(
            ax,
            1.0,
            length,
            pos_center,
            rotation_matrix,
            color="red",
        )

    def _validate_joint_num(self, idx) -> None:
        max_idx = self._param.get_num_links() - 1
        if not 0 <= idx <= max_idx:
            raise ValueError(f"i must be in range [0, num_links{max_idx}]")
