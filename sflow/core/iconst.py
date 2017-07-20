# -*- coding: utf-8 -*-
import os

class _Const(object):
    import math

    floatx = 'float32'
    intx = 'int32'
    eps = 10e-8
    pi = math.pi

const = _Const()

floatx = const.floatx
intx = const.intx
pi = const.pi

