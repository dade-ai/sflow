# -*- coding: utf-8 -*-
import sflow.core as tf
import contextlib
# region hinge margins and variants


@tf.op_scope
def hinge(logits, tsign):
    """
    standard binary hinge loss
    :param logits: logit with signed,
    :param tsign: +-1 assumed,
    :return: (0, inf) loss values
    """
    # same to :
    # return jt.maximum(0., 1. - tsign * psign)
    return tf.nn.relu(1. - tsign * logits)


@tf.op_scope
def hinge_huber(logits, tsign):
    # https://en.wikipedia.org/wiki/Hinge_loss
    # modified huber
    ty = logits * tsign

    # return tf.select(ty.less(-1.), -4. * ty, tf.nn.relu(1. - ty).square())
    return tf.where(ty.less(-1.), -4. * ty, tf.nn.relu(1. - ty).square())


@tf.op_scope
def hinge_rennie(logits, tsign):
    """
    :param logits: logit
    :param target:
    :return:
    """
    # https://en.wikipedia.org/wiki/Hinge_loss
    # todo : make more readible or add some helper
    ty = logits * tsign  # ty: tf.Tensor
    # losses = tf.select(ty.less_equal(0.), 0.5 - ty,
    #                    tf.select(ty.less_equal(1.), (1 - ty).square() * 0.5, 0.))
    losses = tf.where(ty.less_equal(0.), 0.5 - ty,
                       tf.where(ty.less_equal(1.), (1 - ty).square() * 0.5, 0.))

    return losses


@tf.op_scope
def hinge_quad(logits, tsign, gamma=2.):
    """
    quadratically smoothed version by Zhang
    1/(2gamma) x max(0, 1-ty)^2
    https://en.wikipedia.org/wiki/Hinge_loss
    :param logits: logit
    :param target:
    :return:
    """
    # https://en.wikipedia.org/wiki/Hinge_loss
    return 0.5 / float(gamma) * tf.nn.relu(1 - tsign * logits).square()


# endregion

# region hinge various common input types
def hinge_binary(logits, target, hingefn=hinge):
    """
    :param logits: logit with signed, [batch x 1] considered
    :param target: (0, 1) assumed, [batch x 1] considered
    :param hingefn: hinge function one of (hinge, hinge_quad, hinge_huber, hinge_rennie)
    :return: (0, inf) loss value
    """
    # return jt.maximum(0., 1. - tsign * psign).sum()
    ones = tf.ones_like(target)
    tsign = 2. * target - ones
    # following is same to :
    # return hinge_loss(logits, tsign)
    # relu(1 - ty) == maximum(0, 1 - ty)
    # return jt.relu(ones - logits * tsign)
    return hingefn(logits, tsign)


def hinge_index(logits, itarget, hingefn=hinge, axis=-1):
    # assert logits.ndim == itarget.ndim + 1

    depth = logits.dims[axis]
    # make mask for incorrect scores
    binary = tf.one_hot(itarget, depth, axis=axis)
    return hinge_binary(logits, binary, hingefn=hinge)

# endregion

# region hinge binary loss


@tf.op_scope
def hinge_binary_loss(logits, target, hingefn=hinge, axis=-1):
    # assert logits.ndim == target.ndim
    # if target.dims[axis] == 1:
    if _need_to_onehot(logits.dims, target.dims, axis=axis):
        return hinge_index(logits, target, hingefn=hingefn, axis=axis)
    else:
        return hinge_binary(logits, target, hingefn=hingefn).sum()


@tf.op_scope
def hinge_quad_loss(logits, tsign, gamma=2.):
    """
    quadratically smoothed version by Zhang
    1/(2gamma) x max(0, 1-ty)^2
    https://en.wikipedia.org/wiki/Hinge_loss
    :param logits: signed logit score
    :param tsign: +-1 assumed.
    :param gamma:
    :return:
    """
    # sum(.5/gamma * max(0., 1 - target * pred)**2)
    return tf.nn.relu(1 - tsign * logits).l2_loss() / gamma

# endregion

# region hinge for multiclass margin


@tf.op_scope
def hinge_multiclass_margin(logits, itarget, axis=-1):
    """
    Li=∑j≠yi[max(0, wTj * xi − wTyi * xi + 1)] (in case of linear function)

    margins = max(0, scores - correct score + 1)
    margins[target] = 0
    lossi = sum(margins)

    Li = sum(max(0, incorrect scores - correct score + 1))
    L = mean(Li)
    # http://cs231n.github.io/optimization-1/
    # http://cs231n.stanford.edu/slides/winter1516_lecture3.pdf
    :param logits: scores [batch, ... depth], ndim(logits) == ndim(itarget) + 1
    :param itarget: target index [batch, ..., indices]
    :param axis:
    :return:
    """
    assert logits.ndim == itarget.ndim and itarget.dims[axis] == 1

    depth = logits.dims[axis]
    # make mask for incorrect scores
    correct = tf.one_hot(itarget, depth, axis=axis)
    incorrect = 1. - correct

    # dim-reduced correct scores only
    correct_score = (correct * logits).sum(axis=axis, keepdims=True)
    # manual broadcasting
    correct_score = correct_score.repeat(depth, axis=axis)

    # broadcast margins
    # and make margin of target index to zeros
    margins = tf.nn.relu(1. + logits - correct_score) * incorrect

    return margins


@tf.op_scope
def hinge_multiclass_maxmargin(logits, itarget, axis=-1):
    """
    l = max(0, 1 + max(t!=y)(wt*x) - wy*x)
    https://en.wikipedia.org/wiki/Hinge_loss#Optimization
    dade: indirect caclulation using mask. calculate
    l = max(0, 1 + max(incorrect score) - correct score)
    margins = max(0, 1 + scores - correct score)
    margins[correct] = 0
    margins.max(axis) == max(0, 1 + max(incorrect score) - correct score)
    :param logits:
    :param itarget:
    :param axis:
    :return:
    """
    margins = hinge_multiclass_margin(logits, itarget, axis=axis)
    # reduce_max margins
    return margins.max(axis=axis)


@tf.op_scope
def hinge_multiclass_loss(logits, itarget, axis=-1):
    margins = hinge_multiclass_margin(logits, itarget, axis=axis)

    return margins.sum(axis=axis).mean()


@tf.op_scope
def hinge_multiclass_maxmargin_loss(logits, itarget, axis=-1):
    margins = hinge_multiclass_maxmargin(logits, itarget, axis=axis)
    return margins.sum()

# endregion

# region binary cross_entropy

def _need_to_onehot(pdim, tdim, axis):
    if len(pdim) == len(tdim):
        return pdim[axis] != tdim[axis]
    elif len(pdim) == len(tdim) + 1:
        return True
    else:
        # check dimension
        raise ValueError


@tf.op_scope
def sigmoid_cross_entropy(sig, target, axis=-1, eps=1e-8, name=None):

    loss = -target * tf.log(sig + eps) - (1.0 - target) * tf.log(1.0 - sig + eps)
    return tf.identity(loss, name=name or 'sigmoid_cross_entropy')


@tf.op_scope
def binary_cross_entropy(logits, target, axis=-1, name=None):
    """
    if ndim(target) + 1 == ndim(logits):
        # it seem's target is a kind of index
        # maybe a mutual-exclusive classification target
    :param logits: logits
    :param target: ndim(target) == ndim(logits) or ndim(target) + 1 == ndim(logits)
    :return: as is result
    :param axis:
    """
    # change.... check.. logic
    if _need_to_onehot(logits.dims, target.dims, axis):
        target = tf.one_hot(target, logits.dims[-1], axis=axis)

    assert logits.ndim == target.ndim
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=target, name=name)


# endregion

# region softmax and cross entropy

@tf.op_scope
def softmax(t, axis=-1, name=None):
    """ softmax with multi dimension """
    t = tf.exp(t - t.max(axis=axis, keepdims=True))
    s = t.sum(axis=axis, keepdims=True)
    return tf.divide(t, s, name=name)


@tf.op_scope
def hardmax(t, axis=-1, scale=100., name=None):
    """ softmax with multi dimension """
    t *= scale
    t = tf.exp(t - t.max(axis=axis, keepdims=True))
    s = t.sum(axis=axis, keepdims=True)
    return tf.divide(t, s, name=name)


@tf.op_scope
def cross_entropy(p, labels, name=None):
    """
    target.dtype == int, if ndim(p) - 1 = ndim(target), look by index
    cross_entropy(softmax(logits), labels) == softmax_cross_entropy(logits, labels)
    :param p: p == softmax(logits)
    :param labels: index for mutual exclusive target class
    :return:
    """
    return -tf.lookup(p, labels).log(name=name)


@tf.op_scope
def softmax_cross_entropy(logits, labels, name=None):
    """
    with logits!! if with softmax use cross_entropy function

    if ndim(t) == ndim(labels)
        then tf.nn.softmax_cross_entropy_with_logits
    if ndim(t) == ndim(labels)+1, or dims(labels)[axis] == 1
        then sparse_softmax_cross_entropy_with_logits
    support multi dimensional softmax
    :param logits:
    :param labels:
    :param name:
    :return: ndimensial softmax
    """
    if logits.ndim == labels.ndim and labels.dims[-1] == 1:
        labels = labels.squeeze(axis=-1)
    if logits.ndim == labels.ndim + 1:
        # assert labels is a kind of integer
        return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name=name)
    elif logits.ndim == labels.ndim and logits.ndim >= 2:
        # # make ndim to 2
        # assert not [i for i, j in zip(logits.dims, labels.dims) if i != j]
        # dims = logits.dims
        # logits = tf.reshape(logits, [-1, dims[-1]])
        # labels = tf.reshape(labels, [-1, dims[-1]])
        #
        # ce = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
        # return tf.reshape(ce, dims[:-1])
        # r0.12~ master
        return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, dim=-1, name=name)
    else:
        raise ValueError


# endregion

# region sparsemax

@tf.op_scope
def sparsemax(logits, axis=-1, name=None):
    """

    :param logits: tf.Tensor
    :param axis:
    :param name:
    :return:
    """
    # https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/contrib/sparsemax/python/ops/sparsemax.py

    logits = tf.shiftdim(logits, -axis - 1)
    # lshape = logits.shape
    tshape = tf.shape(logits)

    dims = tshape[axis]
    logits = tf.reshape(logits, (-1, dims))
    obs = tf.shape(logits)[0]

    # sort z
    z = logits - tf.mean(logits, axis=1, keepdims=True)
    z_sorted = tf.sort(z)

    # calculate k(z)
    z_cumsum = tf.cumsum(z_sorted, axis=1)
    k = tf.range(1, dims + 1).astype(dtype=logits.dtype)

    z_check = 1 + k * z_sorted > z_cumsum

    k_z = tf.sum(z_check.astype(tf.int32), axis=1)

    # calculate tau(z)
    indices = tf.stack([tf.range(0, obs), k_z - 1], axis=1)
    tau_sum = tf.gather_nd(z_cumsum, indices)
    tau_z = (tau_sum - 1) / k_z.astype(logits.dtype)

    res = tf.maximum(tf.cast(0, logits.dtype), z - tau_z[:, tf.newaxis])

    # rotate axis
    res = tf.reshape(res, tshape)
    # res.set_shape(lshape)
    res = tf.shiftdim(res, axis + 1)

    return res


@tf.op_scope
def sparsemax_loss(logits, sparsemax, labels, axis=-1, name=None):
    # https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/contrib/sparsemax/python/ops/sparsemax_loss.py
    # cf) tf.contrib.sparsemax.sparsemax_loss(logits, sparsemax, labels, name=name)

    shifted_logits = logits - tf.mean(logits, axis=axis, keepdims=True)
    # sum over support
    support = tf.cast(sparsemax > 0, sparsemax.dtype)
    sum_s = support * sparsemax * (shifted_logits - 0.5 * sparsemax)

    # - z_k + ||q||^2
    q_part = labels * (0.5 * labels - shifted_logits)

    return tf.sum(sum_s + q_part, axis=axis)


# endregion


# todo softmax metric or prediction?

# region etc

@tf.op_scope
def top1(logits, target, name=None):
    """
    add example and doc
    :param logits:
    :param target:
    :param name:
    :return:
    """
    topvalue, top_1 = logits.top_k(1)
    top_1 = top_1.squeeze()
    target = target.squeeze()
    assert top_1.ndim == target.ndim

    return top_1.equal(target).to_float().mean(name=name)

# endregion


# region distance

@tf.op_scope
def l2(x, y=None, axis=None, keepdims=False, name=None):
    """
    sum((x - y)^2) or sum(x^2)
    :param x:
    :param y: None | Tensor
    :param axis:
    :param keepdims:
    :param name:
    :return:
    """
    if y is None:
        d = x
    else:
        d = x - y

    s = tf.square(d).sum(axis=axis, keep_dims=keepdims, name=name)

    # try:
    #     s = tf.reduce_sum(tf.square(d), axis=axis, keepdims=keepdims, name=name)
    # except TypeError as e:
    #     s = tf.reduce_sum(tf.square(d), axis=axis, keep_dims=keepdims, name=name)

    return s


@tf.op_scope
def l2mean(x, y=None, axis=None, keepdims=False, name=None):
    """
    sum((x - y)^2) or sum(x^2)
    :param x:
    :param y: None | Tensor
    :param axis:
    :param keepdims:
    :param name:
    :return:
    """
    if y is None:
        d = x
    else:
        d = x - y

    s = tf.reduce_mean(tf.square(d), axis=axis, keepdims=keepdims, name=name)
    return s


@tf.op_scope
def l1(x, y=None, axis=None, keepdims=False, name=None):
    """
    sum(abs(x - y)) or sum(abs(x))
    :param x:
    :param y: None | Tensor
    :param axis:
    :param keepdims:
    :param name:
    :return:
    """
    if y is None:
        d = x
    else:
        d = x - y

    s = tf.reduce_sum(tf.abs(d), axis=axis, keepdims=keepdims, name=name)
    return s

# endregion

