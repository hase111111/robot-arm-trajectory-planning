
#-*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php

# [説明]
# 計算用の関数をまとめたモジュール

import numpy as np

def clamp_angle(angle: float) -> float:
    '''
    角度を -π から π の範囲にクランプする

    Parameters
    ----------
    angle : float
        クランプする角度

    Returns
    -------
    float
        クランプされた角度
    '''
    
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle
