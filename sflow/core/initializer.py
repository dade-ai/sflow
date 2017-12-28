# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
import tensorflow as tf

from .icore import const
from .defaults import default_deco


def _conv_fans(kshape):
    # assert len(kshape) == 4
    # kernel (row, col, inchannel, outchannel) if conv2d

    # if 2d
    # insize = kshape[0] * kshape[1] * kshape[2]  # row x col x indim
    # outsize = kshape[0] * kshape[1] * kshape[3]  # row x col x outdim
    nd = len(kshape) - 2
    kernel_size = np.prod(kshape[:nd])

    # return (insize, outsize)
    return kernel_size * kshape[-2], kernel_size * kshape[-1]


def _fans(shape):
    """ return fanin, fanout"""
    ndim = len(shape)
    if ndim == 2:  # linear in x out
        return shape
    elif ndim >= 3:  # assume convolution filter shape conv1d ~ 3d
        return _conv_fans(shape)
    else:
        # no assumption
        sz = np.sqrt(np.prod(shape))
        return sz, sz


@default_deco
def glorot_uniform(shape, scale=1., dtype=const.floatx, **kwargs):
    fanin, fanout = _fans(shape)
    s = tf.sqrt(6. * scale / (fanin + fanout)).astype(dtype)
    _ = kwargs.pop('partition_info', None)
    return tf.random_uniform_initializer(minval=-s, maxval=s, dtype=dtype, **kwargs)


@default_deco
def glorot_normal(shape, scale=1., dtype=const.floatx, **kwargs):
    fanin, fanout = _fans(shape)
    s = tf.sqrt(3. * scale / (fanin + fanout))
    _ = kwargs.pop('partition_info', None)
    return tf.random_normal_initializer(mean=0.0, stddev=s, dtype=dtype, **kwargs)


@default_deco
def glorot_normaltr(shape, scale=1., dtype=const.floatx, **kwargs):
    fanin, fanout = _fans(shape)
    s = tf.sqrt(3. * scale / (fanin + fanout)).astype(dtype)
    _ = kwargs.pop('partition_info', None)
    return tf.truncated_normal_initializer(mean=0.0, stddev=s, dtype=dtype, **kwargs)


@default_deco
def he_uniform(shape, scale=1., dtype=const.floatx, **kwargs):
    fanin, fanout = _fans(shape)
    s = tf.sqrt(2. * scale / fanin).astype(dtype)

    # fixme : why? and what? how to..
    partition_info = kwargs.pop('partition_info', None)
    if partition_info is not None:
        print('partition_info =? {0}'.format(partition_info))

    return tf.random_uniform_initializer(minval=-s, maxval=s, dtype=dtype, **kwargs)


def eye(shape, dtype=const.floatx, batch_shape=None, name=None):
    # assert len(shape) == 2
    # assert shape[0] == shape[1]
    if isinstance(shape, int):
        return tf.eye(shape, batch_shape=batch_shape, dtype=dtype, name=name)
    if isinstance(shape, (tuple, list)):
        if len(shape) == 2:
            return tf.eye(shape[0], shape[1], batch_shape=batch_shape, dtype=dtype, name=name)
        elif len(shape) == 3:
            assert batch_shape is None
            return tf.eye(shape[1], shape[2], batch_shape=shape[:1], dtype=dtype, name=name)
        else:
            raise ValueError('assert shape.ndim <= 3')
    else:
        raise NotImplementedError('tensor version need?')



# tf.contrib.layers.xavier_initializer
# tf.contrib.layers.xavier_initializer_conv2d
# tf.contrib.layers.variance_scaling_initializer

