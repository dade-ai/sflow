# -*- coding: utf-8 -*-
from tensorflow.python.ops.image_ops_impl import ResizeMethod

import sflow.core as tf
from sflow.core import layer
import numpy as np

# region arg helper


def _kernel_shape(nd, k, indim, outdim):
    if isinstance(k, int):
        k = [k for _ in range(nd)]
    k = list(k)
    assert len(k) == nd
    k.extend([indim, outdim])
    return k


def _stride_shape(nd, s):
    """

    :param nd:
    :param s: int | list | tuple
    :return:
    """
    if isinstance(s, int):
        s = [s for _ in range(nd)]
    s = list(s)
    assert len(s) == nd
    s = [1] + s + [1]
    return s

# endregion

# region conv


# @layer
# @patchmethod(tf.Tensor, tf.Variable)

@layer
def conv(x, outdim, kernel=3, stride=1, pad=0, padding='SAME', mode='CONSTANT',
         initializer=tf.he_uniform, bias=False, **kwargs):
    nd = x.ndim
    if nd == 3:
        return conv1d(x, outdim, kernel, stride=stride, pad=pad, padding=padding, mode=mode,
                      initializer=initializer, bias=bias, **kwargs)
    elif nd == 4:
        return conv2d(x, outdim, kernel, stride=stride, pad=pad, padding=padding, mode=mode,
                      initializer=initializer, bias=bias, **kwargs)

    elif nd == 5:
        return conv3d(x, outdim, kernel, stride=stride, pad=pad, padding=padding, mode=mode,
                      initializer=initializer, bias=bias, **kwargs)
    else:
        raise ValueError('conv for {nd}? nd <= 5'.format(nd=nd))


@layer
def conv1d(x, outdim, kernel, stride=1, pad=0, padding='SAME', mode='CONSTANT',
           initializer=tf.he_uniform, bias=False, **kwargs):

    kernel = _kernel_shape(1, kernel, x.dims[-1], outdim)

    pads = None
    if padding == 'SAME' and mode != 'CONSTANT':
        # pad manually
        half = (kernel[0] - 1) // 2
        pads = [(0, 0), (pad + half, pad + kernel[0] - 1 - half), (0, 0)]
        padding = 'VALID'  # change to valid because manually padded
    elif pad:
        pads = [(0, 0), (pad, pad), (0, 0)]
    if pads is not None:
        x = tf.pad(x, pads, mode=mode)

    W = tf.get_weight('W', shape=kernel, initializer=initializer(kernel), **kwargs)
    out = tf.nn.conv1d(x, W, stride, padding)
    if bias:
        b = tf.get_bias('b', shape=(outdim,), initializer=tf.zeros_initializer(), **kwargs)
        out = tf.nn.bias_add(out, b)

    return out


@layer
def conv2d(x, outdim, kernel, stride=1, pad=0, padding='SAME', mode='CONSTANT',
           initializer=tf.he_uniform, bias=False, **kwargs):

    kernel = _kernel_shape(2, kernel, x.dims[-1], outdim)
    stride = _stride_shape(2, stride)

    pads = None
    if padding == 'SAME' and mode != 'CONSTANT':
        # pad manually
        half = ((kernel[0] - 1) // 2, (kernel[1] - 1) // 2)
        pads = [(0, 0),
                (pad + half[0], pad + kernel[0] - 1 - half[0]),
                (pad + half[1], pad + kernel[1] - 1 - half[1]), (0, 0)]
        padding = 'VALID'  # change to valid because manually padded
    elif pad:
        pads = [(0, 0), (pad, pad), (pad, pad), (0, 0)]
    if pads is not None:
        x = tf.pad(x, pads, mode=mode)

    W = tf.get_weight('W', shape=kernel, initializer=initializer(kernel), **kwargs)
    out = tf.nn.conv2d(x, W, stride, padding)
    if bias:
        b = tf.get_bias('b', shape=(outdim,), initializer=tf.zeros_initializer(), **kwargs)
        out = tf.nn.bias_add(out, b)

    return out


@layer
def conv3d(x, outdim, kernel, stride=1, pad=0, padding='SAME', mode='CONSTANT',
           initializer=tf.he_uniform, bias=False, **kwargs):

    kernel = _kernel_shape(3, kernel, x.dims[-1], outdim)
    stride = _stride_shape(3, stride)  # stride 5-dim

    pads = None
    if padding == 'SAME' and mode != 'CONSTANT':
        # pad manually
        half = ((kernel[0] - 1) // 2, (kernel[1] - 1) // 2, (kernel[2] - 1) // 2)
        pads = [(0, 0),
                (pad + half[0], pad + kernel[0] - 1 - half[0]),
                (pad + half[1], pad + kernel[1] - 1 - half[1]),
                (pad + half[2], pad + kernel[2] - 1 - half[2]), (0, 0)]
        padding = 'VALID'  # change to valid because manually padded
    elif pad:
        pads = [(0, 0), (pad, pad), (pad, pad), (pad, pad), (0, 0)]
    if pads is not None:
        x = tf.pad(x, pads, mode=mode)

    W = tf.get_weight('W', shape=kernel, initializer=initializer(kernel), **kwargs)
    out = tf.nn.conv3d(x, W, stride, padding)
    if bias:
        b = tf.get_bias('b', shape=(outdim,), initializer=tf.zeros_initializer(), **kwargs)
        out = tf.nn.bias_add(out, b)

    return out


# endregion

# region normalization


@layer
def bn(x, stddev=0.002, beta=0.0, gamma=1.0, epsilon=1e-5, decay=0.9, axis=-1, training=None, **kwargs):
    init_gamma = tf.random_normal_initializer(mean=gamma, stddev=stddev)
    init_beta = tf.constant_initializer(beta)

    reuse = tf.get_variable_scope().reuse
    if training is None and (reuse or kwargs.get('reuse', False)):
        training = False
    elif training is None:
        training = x.graph.is_training

    # reuse = reuse is None or reuse is True
    out = tf.layers.batch_normalization(x, axis=axis, momentum=decay, epsilon=epsilon,
                                        beta_initializer=init_beta,
                                        gamma_initializer=init_gamma,
                                        moving_mean_initializer=tf.zeros_initializer(),
                                        moving_variance_initializer=tf.ones_initializer(),
                                        training=training,
                                        **kwargs
                                        )
    return out


@layer
def bn_center(x, stddev=0.002, beta=0.0, gamma=1.0, epsilon=1e-5, decay=0.9, axis=-1, training=None, **kwargs):
    init_gamma = tf.random_normal_initializer(mean=gamma, stddev=stddev)
    # init_beta = tf.constant_initializer(beta)

    reuse = tf.get_variable_scope().reuse
    if training is None and (reuse or kwargs.get('reuse', False)):
        training = False
    elif training is None:
        training = x.graph.is_training

    # reuse = reuse is None or reuse is True
    out = tf.layers.batch_normalization(x, axis=axis, momentum=decay, epsilon=epsilon,
                                        center=True,
                                        # beta_initializer=init_beta,
                                        gamma_initializer=init_gamma,
                                        moving_mean_initializer=tf.zeros_initializer(),
                                        moving_variance_initializer=tf.ones_initializer(),
                                        training=training,
                                        **kwargs
                                        )
    return out


# @layer
def bn_old_buggy(x, stddev=0.002, beta=0.0, gamma=1.0, epsilon=1e-5, decay=0.9, **kwargs):
    #  http://arxiv.org/abs/1502.03167
    # contrib/layers/python/layers/layers.py
    # a = tf.contrib.layers.batch_norm
    # from tensorflow.contrib.layers.python.layers.layers import batch_norm

    # see ioptimizer

    init_gamma = tf.random_normal_initializer(mean=gamma, stddev=stddev)
    init_beta = tf.constant_initializer(beta)

    shapelast = x.dims[-1:]

    # params
    beta_v = tf.get_weight(name='beta', shape=shapelast, initializer=init_beta)
    gamma_v = tf.get_weight(name='gamma', shape=shapelast, initializer=init_gamma)

    # non trainable params
    # collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.MOVING_AVERAGE_VARIABLES]
    collections = [tf.GraphKeys.MODEL_VARIABLES, tf.GraphKeys.MOVING_AVERAGE_VARIABLES]

    moving_mean = tf.get_variable(name='moving_mean', shape=shapelast,
                                  initializer=tf.zeros_initializer(),
                                  trainable=False,
                                  collections=collections)
    moving_var = tf.get_variable(name='moving_var', shape=shapelast,
                                 initializer=tf.ones_initializer(),
                                 trainable=False,
                                 collections=collections)

    out = _batch_normalization(x, beta_v, gamma_v, moving_mean, moving_var, epsilon,
                               decay=decay, **kwargs)

    return out


# buggy
def _batch_normalization(x, beta, gamma, moving_mean, moving_var, epsilon, decay=0.999,
                         axis=None, name=None, is_training=None):
    from tensorflow.python.training import moving_averages

    axis = range(x.ndim - 1) if axis is None else axis  # [0,1,2] for NHWC

    def update_moments():
        m, v = tf.nn.moments(x, axis)
        update_mean = moving_averages.assign_moving_average(moving_mean, m, decay)
        update_var = moving_averages.assign_moving_average(moving_var, v, decay)

        # todo@dade : check this
        with tf.control_dependencies([update_mean, update_var]):
            return tf.identity(m), tf.identity(v)
        # return update_mean, update_var

    if is_training is None:
        training = x.graph.is_training
    else:
        training = is_training

    mean, var = tf.cond(training, update_moments, lambda: (moving_mean, moving_var))

    # todo@dade : check this
    # tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mean)
    # tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, var)

    inference = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon, name=name)
    # inference.set_shape(x.dims)
    inference.set_shape(x.get_shape())

    return inference


@layer
def inorm(x, beta=0.0, gamma=1.0, epsilon=1e-5):
    """
    instance normalization normalization for (W,H)
    same output not regard to trainmode
    # https://arxiv.org/pdf/1607.08022.pdf for instance normalization
    # z = gamma * (x-m)/s + beta
    # note gamma, beta
    :param x: [BHWC] is common case
    :param gamma:
    :param beta:
    :param epsilon:
    :return:
    """
    axes = range(1, 1 + x.ndim-2)  # axes = [1,2] for BWHC except batch, channel
    m, v = tf.nn.moments(x, axes=axes, keep_dims=True)

    # out = (x - m) / tf.sqrt(v + epsilon)
    out = tf.nn.batch_normalization(x, m, v, beta, gamma, epsilon)

    return out


@layer
def cnorm(x, labels, klass=None, stddev=0.01, beta=0.0, gamma=1.0, epsilon=1e-5):
    """
    conditional instance normalization (by label index)
    for learning embedding value of beta and gamma
    # https://arxiv.org/pdf/1610.07629.pdf for conditional instance normalization
    :param x:
    :param labels: [B,]
    :param klass: size of embedding var
    :param gamma: initial_gamma
    :param stddev: stddev for gamma random init
    :param beta: initial beta value
    :param epsilon: 1e-5 for var_epsilon
    :return:
    """
    # total klass count needs !!
    assert klass is not None

    init_gamma = tf.random_normal_initializer(mean=gamma, stddev=stddev)
    init_beta = tf.constant_initializer(beta)
    # params
    shape = [1] * x.ndim
    shape[0] = klass
    shape[-1] = x.dims[-1]  # ones but last channel axis

    # [klass, 1, 1, C] for [BHWC] data
    beta_v = tf.get_weight(name='beta', shape=shape, initializer=init_beta)
    gamma_v = tf.get_weight(name='gamma', shape=shape, initializer=init_gamma)
    # conditioned by label
    # gather
    beta_l = tf.nn.embedding_lookup(beta_v, labels)
    gamma_l = tf.nn.embedding_lookup(gamma_v, labels)

    return inorm(x, beta=beta_l, gamma=gamma_l, epsilon=epsilon)


# @layer
# def pn(x, beta=0.0, gamma=1.0, epsilon=1e-5):
#     b = tf.get_weight(name='beta', shape=(), value=beta)
#     g = tf.get_weight(name='gamma', shape=(), value=gamma)
#
#     return inorm(x, beta=b, gamma=g, epsilon=epsilon)

@layer
def pin(x, beta=0.0, gamma=1.0, epsilon=1e-5):

    shape = [1] * x.ndim
    shape[-1] = x.dims[-1]  # ones but last channel axis

    b = tf.get_weight(name='beta', shape=shape, initializer=tf.constant_initializer(value=beta))
    g = tf.get_weight(name='gamma', shape=shape, initializer=tf.constant_initializer(value=gamma))

    return inorm(x, beta=b, gamma=g, epsilon=epsilon)


@layer
def pbn(x, beta=0.0, gamma=1.0, epsilon=1e-5):

    shape = [1] * x.ndim
    shape[-1] = x.dims[-1]  # ones but last channel axis
    axes = range(x.ndim - 1)  # axes = [1,2] for BWHC except batch, channel

    b = tf.get_weight(name='beta', shape=shape, initializer=tf.constant_initializer(value=beta))
    g = tf.get_weight(name='gamma', shape=shape, initializer=tf.constant_initializer(value=gamma))

    m, v = tf.nn.moments(x, axes=axes, keep_dims=True)

    # out = (x - m) / tf.sqrt(v + epsilon)
    out = tf.nn.batch_normalization(x, m, v, beta, gamma, epsilon)

    return out


@layer
def dn(x, stddev=0.002, beta=0.0, gamma=1.0, epsilon=1e-5, decay=0.9, training=None, **kwargs):
    init_gamma = tf.random_normal_initializer(mean=gamma, stddev=stddev)
    init_beta = tf.constant_initializer(beta)

    shapelast = x.dims[-1:]

    # params
    beta_v = tf.get_weight(name='beta', shape=shapelast, initializer=init_beta)
    gamma_v = tf.get_weight(name='gamma', shape=shapelast, initializer=init_gamma)

    # non trainable params
    # collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.MOVING_AVERAGE_VARIABLES]
    collections = [tf.GraphKeys.MODEL_VARIABLES, tf.GraphKeys.MOVING_AVERAGE_VARIABLES]

    moving_mean = tf.get_variable(name='moving_mean', shape=shapelast,
                                  initializer=tf.zeros_initializer(),
                                  trainable=False,
                                  collections=collections)
    moving_var = tf.get_variable(name='moving_var', shape=shapelast,
                                 initializer=tf.ones_initializer(),
                                 trainable=False,
                                 collections=collections)

    out = _data_normalization(x, beta_v, gamma_v, moving_mean, moving_var, epsilon,
                              decay=decay, training=None, **kwargs)

    return out


def _data_normalization(x, beta, gamma, moving_mean, moving_var, epsilon, decay=0.999,
                        axis=None, training=None, name=None, **kwargs):
    from tensorflow.python.training import moving_averages

    # todo : test

    axis = range(x.ndim - 1) if axis is None else axis  # [0,1,2] for NHWC

    def update_moments():
        m, v = tf.nn.moments(x, axis)
        update_mean = moving_averages.assign_moving_average(moving_mean, m, decay)
        update_var = moving_averages.assign_moving_average(moving_var, v, decay)

        # todo@dade : check this
        # with tf.control_dependencies([update_mean, update_var]):
        #     return tf.identity(moving_mean), tf.identity(moving_var)

        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_var)

        return tf.identity(moving_mean), tf.identity(moving_var)

    reuse = tf.get_variable_scope().reuse
    if training is None:
        if reuse or kwargs.get('reuse', False):
            training = False
        else:
            training = x.graph.is_training

    if training:
        mean, var = update_moments()
    else:
        mean, var = moving_mean, moving_var

    inference = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon, name=name)
    inference.set_shape(x.get_shape())

    return inference


# endregion


# region dense and dropout


@layer
def dropout(x, keep_prob=0.5, is_training=None, noise_shape=None, seed=None):

    if keep_prob == 1.0:
        return x

    def _dropout():
        return tf.nn.dropout(x, keep_prob, noise_shape, seed)
    if is_training is None:
        is_training = x.graph.is_training
    else:
        is_training = tf.convert_to_tensor(is_training)
    return tf.cond(is_training, _dropout, lambda: x)


@layer
def dense(x, outdim, initializer=tf.glorot_uniform, bias=False, name=None):
    """
    out = dense( shape=shape, init=None, paramset=None)
    :param x: tensor
    :param bias:
    :param outdim: output_size
    :param initializer:
    :param name:
    :return: layer | output | (output, params)
    """
    if x.ndim == 4:
        x = x.flat2d()

    assert x.ndim == 2

    outshape = not isinstance(outdim, int)
    if outshape:
        dim = [-1] + list(outdim)
        outdim = np.prod(outdim)

    shape = [x.dims[-1], outdim]
    W = tf.get_weight('W', shape=shape, initializer=initializer(shape))
    # W = tf.get_weight('W', initializer=initializer(shape))
    out = x.dot(W)
    if bias:
        b = tf.get_bias('b', shape=(outdim,), initializer=tf.zeros_initializer())
        out = tf.nn.bias_add(out, b)

    if outshape:
        # make reshape
        out = out.reshape(dim)

    return tf.identity(out, name=name)

# endregion


@layer
def bias(x, initializer=tf.zeros_initializer, name=None):
    outdim = x.dims[-1]
    b = tf.get_bias('b', shape=(outdim,), initializer=initializer())
    return tf.nn.bias_add(x, b, name=name)


# region pooling


def _pool_kernel_stide(dim, kernel, stride):
    if isinstance(kernel, int):
        kernel = [kernel] * dim
    if isinstance(stride, int):
        stride = [stride] * dim
    assert len(kernel) == dim and len(stride) == dim

    return [1] + list(kernel) + [1], [1] + list(stride) + [1]


@layer
def maxpool(x, kernel=2, stride=None, padding='SAME'):
    nd = x.ndim - 2
    stride = kernel if stride is None else stride
    kernel, stride = _pool_kernel_stide(nd, kernel, stride)
    if nd == 2:
        return tf.nn.max_pool(x, kernel, stride, padding)
    elif nd == 3:
        return tf.nn.max_pool3d(x, kernel, stride, padding)
    else:
        raise ValueError('maxpool support {0}? '.format(nd))


@layer
def maxpool_where(x, kernel, stride=None, pads=None, padding='SAME', keep=None):

    # assume kernel == stride
    assert stride is None and padding == 'SAME'
    stride = kernel
    pooled = maxpool(x, kernel, stride=stride, padding=padding)
    mask = where_pooled(x, pooled, kernel, pads=pads)
    if keep is None:
        return pooled, mask
    else:
        keep.append(mask)
        return pooled


@layer
def where_pooled(x, pooled, kernel=None, pads=None):
    """
    return mask
    :param x:
    :param pooled:
    :param kernel:
    :param pads:
    :return:
    """
    # todo : add 3d support
    assert x.ndim == 4
    import math
    if kernel is None:
        kernel = [math.ceil(float(d) / float(p)) for d, p in zip(x.dims, pooled.zip)]
        repeat = pooled.repeats(kernel, axis=[1, 2])
    elif isinstance(kernel, (tuple, list)):
        repeat = pooled.repeats(kernel, axis=[1, 2])
    else:
        repeat = pooled.repeats([kernel, kernel], axis=[1, 2])

    if pads is not None:
        repeat = repeat.pad(pads, axis=[1, 2])
    # crop need
    dim = x.dims
    sameshaped = repeat[:dim[0], :dim[1], :dim[2], :dim[3]]
    mask = tf.equal(x, sameshaped).to_float()

    return mask


@layer
def unpool_where(x, mask, kernel, padding='SAME'):
    """
    unpool with maxpool mask
    :param x:
    :param mask:
    :param kernel:
    :param padding:
    :return:
    """

    # really not a option yet
    # assert stride is None
    assert padding == 'SAME'

    nd = x.ndim
    if nd == 4:
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
        unpooled = x.repeats(kernel, axis=(1, 2))
    elif nd == 5:
        if isinstance(kernel, int):
            kernel = (kernel, kernel, kernel)
        unpooled = x.repeats(kernel, axis=(1, 2, 3))
    else:
        raise ValueError('unsupported nd {0}'.format(nd))

    return unpooled * mask


@layer
def unpool_zero(x, kernel):
    """ upsample by inserting zeros.. """
    if not isinstance(kernel, (list, tuple)) and isinstance(kernel, int):
        kernel = [kernel] * (x.ndim - 2)

    out = x
    for axis in range(1, x.ndim-2):
        out = out.insert_zero(kernel[axis-1], axis=axis)

    return out


@layer
def unpool_repeat(x, kernel):
    """ upsample by repeating"""
    if not isinstance(kernel, (list, tuple)) and isinstance(kernel, int):
        kernel = [kernel] * (x.ndim - 2)

    return x.repeats(kernel, axis=list(range(1, x.ndim-2)))


@layer
def avgpool(x, kernel, stride, padding='SAME'):
    nd = x.ndim - 2
    kernel, stride = _pool_kernel_stide(nd, kernel, stride)
    if nd == 2:
        return tf.nn.avg_pool(x, kernel, stride, padding)
    elif nd == 3:
        return tf.nn.avg_pool3d(x, kernel, stride, padding)
    else:
        raise ValueError('avgpool support {0}? '.format(nd))


@layer
def gpool(x, keepdims=True):
    """
    global_avgpool
    :param x:
    :param keepdims:
    :return:
    """
    # http://arxiv.org/pdf/1312.4400.pdf
    axis = list(range(1, x.ndim-1))
    return x.mean(axis=axis, keepdims=keepdims)


# endregion

# region atrous convolution

# def atrous2d(x, )

def _atrous1d(x, kernel, rate, padding='SAME'):
    """
    cf https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#atrous_conv2d
    :param x: [batch, time, channel]
    :param kernel: [1, 1, inchannel, outchannel]
    :param rate: dialtion rate
    :param padding: 'same' or 'valid'
    :param bias:
    :return:
    """
    # from ireshape import time_to_batch, batch_to_time

    # atrous_conv1d implementation
    if rate == 1:
        # same to normal conv1d
        out = tf.nn.conv1d(x, kernel, stride=(1, 1, 1), padding=padding)
        return out

    # if 'same'
    if padding == 'SAME':
        filter_width = kernel.dims[0]
        # temporal dimension of the filter and the upsampled filter in which we
        # introduce (rate - 1) zeros between consecutive filter values.
        filter_width_up = filter_width + (filter_width - 1) * (rate - 1)
        pad = filter_width_up - 1

        # When pad is odd, we pad more to right
        pad_left = pad // 2
        pad_right = pad - pad_left
    elif padding == 'VALID':
        pad_left = 0
        pad_right = 0
    else:
        raise ValueError('Invalid padding')

    in_width = x.dims[1] + pad_left + pad_right
    # more padding so that rate divides the width of the input
    pad_right_extra = (rate - in_width % rate) % rate
    pads = [(0, 0), (pad_left, pad_right + pad_right_extra), (0, 0)]

    out = x.time_to_batch(rate, pads)

    out = tf.nn.conv1d(out, kernel, stride=(1, 1, 1), padding='VALID')
    # if bias is not None:
    # bias=bias,

    crops = [(0, 0), (0, pad_right_extra), (0, 0)]

    # temporary test this
    out = out.batch_to_time(rate, crops)

    return out


@layer
def atrous(x, outdim, kernel, rate, pad=0, padding='SAME',
           initializer=tf.he_uniform, bias=None, **kwargs):
    # todo rate per axis?

    assert isinstance(pad, int)
    nd = x.ndim - 2
    if pad:
        pads = [(0, 0)] + [(pad, pad)] * nd + [(0, 0)]
        x = tf.pad(x, pads, mode='CONSTANT')

    kernel = _kernel_shape(nd, kernel, x.dims[-1], outdim)
    W = tf.get_weight('W', shape=kernel, initializer=initializer(kernel), **kwargs)

    if nd == 1:
        out = _atrous1d(x, W, rate, padding=padding)
    elif nd == 2:
        out = tf.nn.atrous_conv2d(x, W, rate, padding)
    else:
        raise NotImplementedError('not implementd for ndim [{0}]'.format(nd))

    if bias is not None:
        b = tf.get_bias('b', shape=(outdim,), initializer=tf.zeros_initializer(), **kwargs)
        out = tf.nn.bias_add(out, b)

    return out


# endregion

# region deconv


def _deconv_outshape(nd, inshape, outdim, kernel, stride, padding, extra_shape=0):
    # conv2d case (filter = kernel)
    # output = (input + stride - 1)//stride       # SAME ? filter?
    # output = (input + stride - filter)//stride   # VALID
    # 위 식 inverse
    # output = (input * stride) - stride + 1  + extra
    # todo : through check need ??
    # => max일경우 (output - 1) * stride + 1 - stride
    # output = (input * stride) - stride + filter + extra  # VALID
    # 단, 0 <= extra < stride
    if isinstance(kernel, int):
        kernel = [kernel] * nd
    if isinstance(stride, int):
        stride = [stride] * nd
    if extra_shape is None:
        extra_shape = 0
    if isinstance(extra_shape, int):
        extra_shape = [extra_shape] * nd

    outshape = [None] * nd
    if padding == 'SAME':
        for i in range(0, nd):
            outshape[i] = inshape[i+1] * stride[i] + extra_shape[0]
    elif padding == 'VALID':
        # assert -stride[0] < extra_shape[0] < stride[0]
        # assert -stride[1] < extra_shape[1] < stride[1]
        for i in range(0, nd):
            outshape[i] = (inshape[i+1] * stride[i]) - stride[i] + kernel[i] + extra_shape[i]
    else:
        raise ValueError('unknown padding option {0}'.format(padding))

    return [inshape[0]] + outshape + [outdim]


@layer
def deconv(x, outdim, kernel, stride=1, padding='SAME',
           initializer=tf.he_uniform, bias=False, extra=None, **kwargs):
    nd = x.ndim - 2
    out_shape = _deconv_outshape(nd, x.dims, outdim, kernel, stride, padding, extra)
    oshape = tf.TensorShape(out_shape)
    if out_shape[0] is None:
        out_shape[0] = tf.shape(x)[0]
        out_shape = tf.stack(out_shape)

    kernel_shape = _kernel_shape(nd, kernel, outdim, x.dims[-1])  # swap in and out channel
    stride = _stride_shape(nd, stride)  # stride

    W = tf.get_weight('W', shape=kernel_shape, initializer=initializer(kernel_shape))

    if nd == 2:
        out = tf.nn.conv2d_transpose(x, W, out_shape, strides=stride, padding=padding)
    elif nd == 3:
        out = tf.nn.conv3d_transpose(x, W, out_shape, strides=stride, padding=padding)
    else:
        raise NotImplementedError('not implementd for ndim [{0}]'.format(nd))

    if bias:
        b = tf.get_bias('b', shape=(outdim,), initializer=tf.zeros_initializer(), **kwargs)
        out = tf.nn.bias_add(out, b)

    out.set_shape(oshape)

    return out


# endregion

# region depthwise

@layer
def dwconv(x, kernel, multiplier=1, stride=1, pad=0, padding='SAME',
           initializer=tf.he_uniform, bias=False, **kwargs):

    if pad:
        pads = [(0, 0), (pad, pad), (pad, pad), (0, 0)]
        x = tf.pad(x, pads, mode='CONSTANT')

    kernel = _kernel_shape(2, kernel, x.dims[-1], multiplier)
    stride = _stride_shape(2, stride)

    W = tf.get_weight('W', shape=kernel, initializer=initializer(kernel), **kwargs)
    out = tf.nn.depthwise_conv2d(x, W, stride, padding)
    if bias:
        outdim = kernel[2] * multiplier
        b = tf.get_bias('b', shape=(outdim,), initializer=tf.zeros_initializer(), **kwargs)
        out = tf.nn.bias_add(out, b)

    return out

# endregion

# region subpixel


@layer
def subpixel(x, kernel, factor=2, stride=1, pad=0, padding='SAME',
             initializer=tf.he_uniform, bias=False, **kwargs):
    from .ireshape import channel_to_space

    assert x.ndim == 4  # implemented for 4D tensor

    indim = x.dims[-1]
    outdim = indim * factor * factor

    kernel = _kernel_shape(2, kernel, indim, outdim)
    stride = _stride_shape(2, stride)

    W = tf.get_weight('W', shape=kernel, initializer=initializer(kernel))
    out = tf.nn.conv2d(x, W, stride, padding=padding)
    if bias:
        b = tf.get_bias('b', shape=(outdim,), initializer=tf.zeros_initializer())
        out = tf.nn.bias_add(out, b)

    # periodic shuffle
    out = channel_to_space(out, factor)

    return out

# endregion


# region activation


@layer
def leaky(x, slope=0.01, name=None):
    """
    leaky_relu
    see also pleaky
    :param x:
    :param slope: 0.01 default
    :return:
    """
    return tf.maximum(x, x*slope, name=name)


@layer
def pleaky(x):
    """
    parametric leakyrelu
    :param x:
    :return:
    """
    alpha = tf.get_bias('alpha', shape=(), initializer=tf.constant_initializer(0.01))
    return tf.maximum(x, x * alpha)

# endregion

# region resize images


@layer
def sizeup(x, factor=(2, 2), extras=(0, 0), method=ResizeMethod.NEAREST_NEIGHBOR, align_corners=False):
    inshape = x.dims
    if isinstance(factor, int):
        factor = (factor, factor)
    if isinstance(extras, int):
        extras = (extras, extras)

    hw = [inshape[1] * factor[0] + extras[0], inshape[2] * factor[1] + extras[1]]
    return tf.image.resize_images(x, hw, method=method, align_corners=align_corners)


@layer
def sizedown(x, factors=(2, 2), extras=(0, 0), method=ResizeMethod.NEAREST_NEIGHBOR, align_corners=False):
    inshape = x.dims
    if isinstance(factors, int):
        factors = (factors, factors)
    if isinstance(extras, int):
        extras = (extras, extras)

    hw = [inshape[1] // factors[0] + extras[0], inshape[2] // factors[1] + extras[1]]

    return tf.image.resize_images(x, hw, method=method, align_corners=align_corners)


# endregion

# region collecting utils

@layer
def keep(t, keepto, collection=None):
    """
    append to list and return t as is
    :param t: tensor
    :param keepto: list
    :return:
    """
    if collection is not None:
        tf.add_to_collection(collection, t)
    keepto.append(t)
    return t


@layer
def collect(t, collection='activation'):
    """
    append to list and return t as is
    :param t: tensor
    :param collection:
    :return:
    """
    tf.add_to_collection(collection, t)
    return t

# endregion

# region util

@layer
def iname(t, name):
    return tf.identity(t, name=name)

# endregion
