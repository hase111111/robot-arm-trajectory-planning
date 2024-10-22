
#-*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php

# [説明]
# ロボットと衝突する障害物を定義するプログラム

import abc

from two_link.two_link_robot import TwoLinkRobot

class Obstacle(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def is_collision(self, robot: TwoLinkRobot) -> bool:
        raise NotImplementedError()
