# -*- coding: utf-8 -*-
import sflow.core as tf


@tf.op_scope
def dice_coeff(pred, target, axis=(1, 2), eps=1e-8):
    """
    dice coeff, ndim == 4, nhwc format
    2*(intersect) / (pred + target)
    :param pred: assume : range(0,1), 4dim
    :param target: 0 or 1 nhwc format 4dim
    :param axis: reduction_axis
    :param eps: for prevent NaN
    :return: [batch x channel]
    """
    ps = pred.sum(axis=axis)
    ts = target.sum(axis=axis)
    intersect = tf.sum(pred * target, axis=axis)

    # need some set_shape?
    # select or add some slack?
    # return jt.select(jt.equal(ps * ts, 0.), jt.ones(shape=ts.dims), 2. * intersect / (ps + ts))

    return (2. * intersect + eps) / (ps + ts + eps)


@tf.op_scope
def dice_coeff_digit(pred, target, axis=None, axis_img=(1,2), threshold=0.2):
    """
    excact dice coeff. assume target value is zero or ones
    :return: [batch x channel]
    """

    ps = pred.sum(axis=axis_img)
    ts = target.sum(axis=axis_img)
    intersect = tf.sum(pred * target, axis=axis_img)

    # gt nomask and exact pred

    inomask = tf.less_equal(pred.max(axis=axis_img), threshold)
    nomask_score = 1. - tf.any(target, axis=axis_img).to_float()
    # dice_batch = tf.select(inomask, nomask_score, 2.*intersect / (ps + ts))
    dice_batch = tf.where(inomask, nomask_score, 2.*intersect / (ps + ts))

    # make wanted reduced_axis
    if axis is None:
        return dice_batch.mean()
    else:
        return dice_batch


@tf.op_scope
def dice_coeff_forward(pred, target, axis=None, axis_img=(1, 2), threshold=0.2):
    """
    excact dice coeff. assume target value is zero or ones
    :param pred: [batch, h, w, 1]
    :param target: [batch, h, w, 1]
    :param axis:
    :param threshold: pixel < 0.2 then pxiel <= 0
    :return: [batch x channel]
    """
    # assert axis is not None
    # pred = tf.select(tf.less_equal(pred, threshold), tf.zeros(shape=pred.dims), tf.ones(shape=pred.dims))
    pred = tf.where(tf.less_equal(pred, threshold), tf.zeros(shape=pred.dims), tf.ones(shape=pred.dims))

    ps = pred.sum(axis=axis_img)
    ts = target.sum(axis=axis_img)
    intersect = tf.sum(pred * target, axis=axis_img)

    # gt nomask and exact pred

    imask = tf.any(pred, axis=axis_img)
    nomask_score = 1. - tf.any(target, axis=axis_img).to_float()
    # dice_batch = tf.select(imask, 2. * intersect / (ps + ts), nomask_score)
    dice_batch = tf.where(imask, 2. * intersect / (ps + ts), nomask_score)

    # make wanted reduced_axis
    if axis is None:
        return dice_batch.mean()
    else:
        return dice_batch.mean(axis)


@tf.op_scope
def dice_loss(pred, target, axis=None):
    return 1.0 - dice_coeff(pred, target, axis=axis)


@tf.op_scope
def log_dice_coeff(pred, target, axis=(1, 2), eps=1e-8):

    ps = pred.sum(axis=axis, keepdims=False)
    ts = target.sum(axis=axis, keepdims=False)
    intersect = tf.sum(pred * target, axis=axis, keepdims=False)

    return 2. * tf.log(intersect + eps) - tf.log(ps + ts + eps)


def coordinate2d(t):
    """
    -1.~1. coordinate of x, y, left top = (-1., -1.)
    :param x:
    :return:
    """
    # fixme to 0.10.0
    import numpy as np
    dims = t.dims
    y, x = 1, 2
    x = np.linspace(-1, 1, dims[x])
    y = np.linspace(-1, 1, dims[y])
    xv, yv = np.meshgrid(x, y)
    xbatch = np.tile(xv, [dims[0], 1, 1]).astype('float32')
    ybatch = np.tile(yv, [dims[0], 1, 1]).astype('float32')

    xgrid = tf.convert_to_tensor(xbatch[..., np.newaxis])
    ygrid = tf.convert_to_tensor(ybatch[..., np.newaxis])

    return xgrid, ygrid

    # tf cumsum 0.10
    # xcoord = jt.ones(shape=t.dims).cumsum(axis=axis[0]) / t.dims[axis[0]]
    # ycoord = jt.ones(shape=t.dims).cumsum(axis=axis[1]) / t.dims[axis[1]]

    # not supported v.0.9.0
    # dims = t.dims
    # x, y = axis[1], axis[0]
    # xcoord = jt.linspace(0.0, 1.0, dims[x]) / dims[x]
    # xcoord = xcoord.reshape((1, dims[x])).repeat(dims[y], axis=0)
    # ycoord = jt.linspace(0.0, 1.0, dims[y]).to_float() / dims[y]
    # ycoord = ycoord.reshape((dims[y], 1)).repeat(dims[x], axis=1)


def center_coord2d(t):
    """ t nhwc format """
    x, y = coordinate2d(t)
    cx = (t * x).mean(axis=range(1, t.ndim))
    cy = (t * y).mean(axis=range(1, t.ndim))

    return tf.pack(cx, cy).T


def center_l2_distance(x, t):
    """
    batch
    :param x: nhw1
    :return:
    """
    return tf.square(center_coord2d(t) - center_coord2d(x))


def l2_center_loss(x, target):
    center = center_coord2d(target)
    xcoord, ycoord = coordinate2d(x)

    return tf.nn.l2_loss(xcoord - center[:, 0]) + tf.nn.l2_loss(ycoord - center[:, 1])

