"""provide type check functions"""

# -*- coding: utf-8 -*-

# Copyright (c) 2024 Taisei Hasegawa
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php


import inspect


def _type_checked(val, type_, name=None):
    if name is None:
        # 呼び出し元のフレームから変数名を取得
        frame = inspect.currentframe().f_back
        for var_name, var_val in frame.f_locals.items():
            if var_val is val:
                name = var_name
                break
        # リテラルの場合、値そのものをnameに使う
        if name is None:
            name = repr(val)
    if not isinstance(val, type_):
        raise TypeError(f"{name} must be {type_.__name__}, not {type(val).__name__}")
    return val
