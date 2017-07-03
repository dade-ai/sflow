# -*- coding: utf-8 -*-
import tensorflow as tf
from .iconst import const


def multi_hot(indices, klasses, dtype=const.floatx, axis=-1, name=None, **kwargs):
    """

    :param indices: [batch, klasses]
    :param klasses: num of klasses, assert len(assert sum(klasses) == indices.dims[axis], assert sum(klasses) == res.dims[axis]
    :param dtype:
    :param axis:
    :param name:
    :param kwargs:
    :return:
    """
    # tf.slice(input_, begin, size, name=None)
    if axis != -1 and axis != indices.ndim:
        raise ValueError('need support?')

    assert len(klasses) == indices.dims[axis]
    assert indices.ndim == 2

    onehots = [tf.one_hot(indices[:, i], klasses[i], dtype=dtype, **kwargs)
               for i in range(indices.dims[axis])]
    mhot = tf.concat(onehots, axis=axis, name=name)

    return mhot
