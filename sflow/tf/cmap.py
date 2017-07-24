# -*- coding: utf-8 -*-
from snipy.plt import cmap
import tensorflow as tf

# see : http://matplotlib.org/api/cm_api.html?highlight=norm%20colormap

_funs = [f for f in dir(cmap) if not f.startswith('_')]


def _tf_cmap(fname):
    pyf = cmap.__dict__[fname]

    def wrapper(t, alpha=None, bytes=None, lut=None, dtype=tf.float32):
        def _pyf(x):
            return pyf(x, alpha=alpha, bytes=bytes, lut=lut)
        if t.dims[-1] == 1:
            t = t.squeeze(-1)
        res = tf.py_func(_pyf, [t], tf.double).astype(dtype)
        res.set_shape(t.dims[:3] + [4])
        return res

    return wrapper

# example implementatation


def jet(t, alpha=None, bytes=None, lut=None, dtype=tf.float32):
    """
    example:
        tf.cmap.jet(t, alpha=None, bytes=bytes, lut=None, dtype=tf.float32)

    :param t:
    :param alpha:
    :param bytes:
    :param lut: If lut is not None it must be an integer giving the number of entries desired in the lookup table,
    and name must be a standard mpl colormap name.
    :param dtype:
    :return:
    """
    binded = lambda x: cmap.jet(x, alpha=alpha, bytes=bytes, lut=lut)
    if t.dims[-1] == 1:
        t = t.squeeze(-1)
    res = tf.py_func(binded, [t], tf.double).astype(dtype)
    # fixme
    res.set_shape(res.set_shape(t.dims[:3] + [4]))
    return res

locals().update({n: _tf_cmap(n) for n in _funs})

# example
# tf.cmap.Greys(t, alphas=None, bytes=bytes, lut=None, dtype=tf.float32)

