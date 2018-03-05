# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from snipy.basic import (patch, patchmethod, patchproperty, tuple_args)
from .iconst import const
from .icontext import op_scope


# todo check v1.12.0
_pyall = all


_common = [
    # shapes
    tf.size, tf.rank, tf.squeeze, tf.expand_dims,  # tf.shape, r.0.13 property?

    # slicing and joining
    tf.tile, tf.reverse_sequence, tf.reverse, tf.reverse_v2,
    # slightly modified tf.unstack, tf.unpack, tf.split, tf.split_v
    tf.space_to_batch, tf.batch_to_space_nd, tf.batch_to_space,
    tf.space_to_depth, tf.depth_to_space, tf.gather, tf.gather_nd,
    tf.boolean_mask,
    tf.dequantize, tf.quantize_v2, tf.setdiff1d,

    # from matrix math
    tf.matmul,   # tf.batch_matmul, removed from r.0.12.1?

    # Arithmetic operators
    tf.add, tf.subtract, tf.multiply, tf.scalar_mul, tf.div, tf.divide,
    tf.truediv, tf.floordiv, tf.realdiv, tf.truncatediv, tf.floor_div,
    tf.truncatemod, tf.floormod, tf.mod, tf.cross,

    # Reduce
    tf.reduce_sum, tf.reduce_mean, tf.reduce_prod,
    tf.reduce_max, tf.reduce_min,  # tf.reduce_all, tf.reduce_any,8

    # Basic Math Functions
    tf.add_n, tf.abs, tf.negative, tf.sign, tf.reciprocal, tf.square,
    tf.round, tf.sqrt, tf.rsqrt, tf.pow, tf.exp, tf.log, tf.log1p, tf.ceil,
    tf.floor, tf.maximum, tf.minimum, tf.cos, tf.sin, tf.lbeta, tf.tan,
    tf.acos, tf.asin, tf.atan, tf.lgamma, tf.digamma, tf.erf, tf.erfc,
    tf.squared_difference, tf.igamma, tf.igammac, tf.zeta, tf.polygamma, tf.betainc, tf.rint,

    # scan
    tf.cumsum, tf.cumprod,

    # segmentation
    tf.segment_sum, tf.segment_prod, tf.segment_min, tf.segment_max, tf.segment_mean,
    tf.argmin, tf.argmax, tf.setdiff1d, tf.unique,

    # comparision
    tf.equal, tf.not_equal, tf.less, tf.less_equal, tf.greater, tf.greater_equal,

    # activations
    tf.nn.relu, tf.nn.relu6, tf.nn.crelu, tf.nn.elu, tf.nn.softplus,

    # classification
    tf.nn.softsign, tf.nn.dropout, tf.nn.bias_add, tf.sigmoid, tf.tanh,
    tf.nn.softmax, tf.nn.log_softmax,
    tf.nn.in_top_k,  # tf.nn.top_k ommitted

    # castings
    tf.string_to_number, tf.to_double, tf.to_float, tf.to_bfloat16,
    tf.to_int32, tf.to_int64, tf.cast, tf.bitcast, tf.saturate_cast,

    tf.clip_by_value,
    # loss
    tf.nn.l2_loss,
    ]

patch.methods([tf.Tensor, tf.Variable], _common)

# function aliases
to_tensor = tf.convert_to_tensor


@patchmethod(tf.Tensor, tf.Variable)
def astype(x, dtype, name=None):
    return tf.cast(x, dtype, name=name)


def _reduce_helper(reduce_fun, *args, **kwargs):
    try:
        return reduce_fun(*args, **kwargs)
    except TypeError:
        if 'keepdims' in kwargs:
            kwargs['keep_dims'] = kwargs['keepdims']
            del kwargs['keepdims']
            return reduce_fun(*args, **kwargs)


@patchmethod(tf.Tensor, tf.Variable)
def sum(x, **kwargs):
    return _reduce_helper(tf.reduce_sum, x, **kwargs)


@patchmethod(tf.Tensor, tf.Variable)
def mean(x, **kwargs):
    return _reduce_helper(tf.reduce_mean, x, **kwargs)


@patchmethod(tf.Tensor, tf.Variable)
def prod(x, **kwargs):
    return _reduce_helper(tf.reduce_prod, x, **kwargs)


@patchmethod(tf.Tensor, tf.Variable)
def max(x, **kwargs):
    return _reduce_helper(tf.reduce_max, x, **kwargs)


@patchmethod(tf.Tensor, tf.Variable)
def min(x, **kwargs):
    return _reduce_helper(tf.reduce_min, x, **kwargs)


@patchproperty(tf.Tensor, tf.Variable)
def ndim(t):
    """ tf.rank()와 동일 하나 tensor가 아니라 value"""
    return len(t.shape)


@patchproperty(tf.Tensor, tf.Variable)
def dims(t):
    """
    shape = list of shape value not tensor
    ex)
    v = tf.variable([1, 2, 3])
    print v.shape
    """
    return t.shape.as_list()  # [1,2,3]


@patchproperty(tf.Tensor, tf.Variable)
def shapes(t):
    """
    shape = list of shape int value or tensor for not determined shape
    ex)
    v = tf.variable([1, 2, 3])
    print v.size
    """
    d = t.dims
    s = tf.shape(t)
    out = [s[i] if d is None else d for i, d in enumerate(t.dims)]
    return out


@patchproperty(tf.Tensor, tf.Variable)
def numel(t):
    ishape = t.get_shape()
    if not ishape.is_fully_defined():
        return None
    return np.prod([d.value for d in ishape])


@patchmethod(tf.Tensor, tf.Variable)
def dot(x, y, **kwargs):
    """
    dot == matmul or batch_matmul
    :param x:
    :param y:
    :return:
    """
    # if x.ndim == 2 and y.ndim == 2:
    #     return tf.matmul(x, y)
    # elif x.ndim == 3 and y.ndim == 3:
    #     return tf.batch_matmul(x, y)  # r.0.13 deprecated?
    # else:
    #     raise ValueError
    return tf.matmul(x, y, **kwargs)


@patchmethod(tf.Tensor, tf.Variable)
@op_scope
def std(x, axis=None, keepdims=False, name=None):
    if x.dtype.base_dtype == tf.bool:
        x = tf.cast(x, const.floatx)
    # in first, keep_dims = True
    m = tf.reduce_mean(x, axis=axis, keepdims=True)
    devs_squared = tf.square(x - m)
    return tf.sqrt(tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims), name=name)


@patchmethod(tf.Tensor, tf.Variable)
@op_scope
def var(x, axis=None, keepdims=False, name=None):
    if x.dtype.base_dtype == tf.bool:
        x = tf.cast(x, const.floatx)
    # in first, keep_dims = True
    m = tf.reduce_mean(x, axis=axis, keepdims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims, name=name)


@patchmethod(tf.Tensor, tf.Variable)
def reduce_any(x, axis=None, keepdims=False, name=None):
    return _reduce_helper(tf.reduce_any, x.cast(tf.bool), axis=axis, keepdims=keepdims, name=name)


@patchmethod(tf.Tensor, tf.Variable)
def reduce_all(x, axis=None, keepdims=False, name=None):
    return _reduce_helper(tf.reduce_all, x.cast(tf.bool), axis=axis, keepdims=keepdims, name=name)


@patchmethod(tf.Tensor, tf.Variable)
def all(x, **kwargs):
    return reduce_all(x, **kwargs)


@patchmethod(tf.Tensor, tf.Variable)
def any(x, **kwargs):
    return reduce_any(x, **kwargs)


def _support_negative_axis(x, axis):
    if axis is not None and axis < 0:
        assert axis >= -x.ndim
        axis %= x.ndim
    return axis


@patchmethod(tf.Tensor, tf.Variable)
def argmax(x, axis=-1, name=None):
    axis = _support_negative_axis(x, axis)
    return tf.argmax(x, axis, name=name)


@patchmethod(tf.Tensor, tf.Variable)
def argmin(x, axis=-1, name=None):
    axis = _support_negative_axis(x, axis)
    return tf.argmin(x, axis, name=name)


def concat(dim, tensors, name=None):
    try:
        return tf.concat_v2(values=tensors, axis=dim, name=name)
    except AttributeError:
        return tf.concat(values=tensors, axis=dim, name=name)


@patchmethod(tf.Tensor, tf.Variable)
def cat0(t, *values, **kwargs):
    return concat(0, [t] + list(values), **kwargs)


@patchmethod(tf.Tensor, tf.Variable)
def cat1(t, *values, **kwargs):
    return concat(1, [t] + list(values), **kwargs)


@patchmethod(tf.Tensor, tf.Variable)
def cat(t, *values, **kwargs):
    return concat(t.ndim - 1, [t] + list(values), **kwargs)


@patchmethod(tf.Tensor, tf.Variable)
def append(t, s, **kwargs):
    return concat(t.ndim - 1, [t, [s]], **kwargs)


@patchmethod(tf.Tensor, tf.Variable)
def reshape(t, *shape, **kwargs):
    """
    t.reshape([...]) or t.reshape(dim0, dim1, ...)
    :param t:
    :param shape:
    :return:
    """
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return tf.reshape(t, shape, **kwargs)


@patchmethod(tf.Tensor, tf.Variable)
def pad(t, paddings, axis=None, mode='CONSTANT', name=None):
    if axis is None:
        return tf.pad(t, paddings, mode=mode, name=name)
    else:
        return padaxis(t, paddings, axis, mode=mode, name=name)


@patchmethod(tf.Tensor, tf.Variable)
def padaxis(t, paddings, axis, mode='CONSTANT', name=None):
    """
    t.pad((1,1), axis=0)  # padleft, padright
    t.pad([(1,1), (1,1)], axis=[0,1])  # padleft, right, top, bottom
    :param t:
    :param paddings:
    :param axis:
    :param mode:
    :return:
    """
    if isinstance(axis, int):
        axis = (axis,)
        if len(paddings) == 2:
            paddings = [paddings]  # ex) pad(t, (1,1), axis=0) == pad(t, [(1,1)], axis=0)
        assert len(axis) == len(paddings)

    assert t.ndim >= len(paddings)

    # axis = list(axis)
    padallaxis = [(0, 0)] * t.ndim
    for i, padding in zip(axis, paddings):
        padallaxis[i] = padding

    return tf.pad(t, padallaxis, mode=mode, name=name)


def pad_to_shape(x, shape=None, name=None):
    """
    :param x: Tensor
    :param shape: list
    :param name:
    :return: shape(tensor) == shape
    """
    # ex) correspondent logic for ND
    # for 1D
    # tf.Assert(tf.shape(u.y)[0] <= size, [u.y])
    # us = tf.pad(u.y, [(0, size - tf.shape(u.y)[0])])

    assert x.ndim == len(shape)

    dims = tf.shape(x)
    padding = [(0, 0) if s is None else (0, s - dims[i])
               for i, s in enumerate(shape)]
    res = tf.pad(x, padding, name=name or 'pad_to')

    res_shape = [d if s is None else s
                 for s, d in zip(shape, x.shape)]
    res.set_shape(res_shape)

    return res


@patchmethod(tf.Tensor, tf.Variable)
def crop(x, crops, axis=None, name=None):
    if axis is not None:
        raise NotImplementedError

    slices = [slice(c0, -c1 or None) for c0, c1 in crops]
    res = x.__getitem__(slices)
    return tf.identity(res, name=name or 'crop')


@patchmethod(tf.Tensor, tf.Variable)
def transpose(x, *perm, **kwperm):
    if len(perm) == 1 and isinstance(perm[0], (tuple, list)):
        perm = perm[0]
    if kwperm:
        perm = kwperm.pop('perm', perm or None)
    name = kwperm.pop('name', 'transpose')
    return tf.transpose(x, perm=perm, name=name)

patch.getter([tf.Tensor, tf.Variable], tf.transpose, 'T')


@patchmethod(tf.Tensor, tf.Variable)
def transpose_inv(x, *perm, **kwargs):
    import numpy as np
    if len(perm) == 1 and isinstance(perm[0], (tuple, list)):
        perm = perm[0]
    pattern = np.argsort(perm)
    return tf.transpose(x, perm=pattern, **kwargs)


@patchmethod(tf.Tensor, tf.Variable)
def shiftdim(t, shift, name=None):
    """
    shift dims to right (or left for negative shift) direction.
    :param t: tensor
    :param shift: int
    :param name:
    :return:
    """
    ndim = t.ndim
    if shift > 0:
        axis = list(range(ndim - shift, ndim)) + list(range(t.ndim - shift))
    elif shift < 0:
        axis = list(range(-shift, ndim)) + list(range(-shift))
    else:
        return t
    return tf.transpose(t, perm=axis, name=name)


@patchmethod(tf.Tensor, tf.Variable)
def flat2d(t, name=None):
    import numpy as np
    if t.dims[0] is not None:
        return tf.reshape(t, [t.dims[0], -1])
    elif _pyall(d for d in t.dims if d is not None):
        return tf.reshape(t, [-1, np.prod(t.dims[1:])])
    else:
        return tf.reshape(t, tf.stack([tf.shape(t)[0], -1]), name=name)


@patchmethod(tf.Tensor, tf.Variable)
def flat(t, name=None):
    return tf.reshape(t, [-1], name=name)


@patchmethod(tf.Tensor, tf.Variable)
def cvec(t, name=None):
    return tf.reshape(t, [-1, 1], name=name)


@patchmethod(tf.Tensor, tf.Variable)
def rvec(t, name=None):
    return tf.reshape(t, [1, -1], name=name)


@patchmethod(tf.Tensor, tf.Variable)
@op_scope
def repeat(x, rep, axis=-1, name=None):
    """
    element repeat along an axis
    :param x:
    :param rep:
    :param axis:
    :return:
    """
    axis = _support_negative_axis(x, axis)
    # indim = x.dims
    multiples = [1] * x.ndim
    multiples[axis] = rep
    data = tf.tile(x, multiples=multiples)

    indim = tf.shape(x)
    indim = tf.unstack(indim)
    indim.insert(axis, rep)

    data = tf.reshape(data, indim)
    # swap axis
    ax = list(range(len(indim)))
    ax[axis] = axis + 1
    ax[axis+1] = axis
    data = tf.transpose(data, perm=ax)

    # merge axis
    del indim[axis]
    indim[axis] *= rep

    out = tf.reshape(data, indim, name=name)

    # set shape information
    indim = x.dims
    if indim[axis] is not None:
        indim[axis] *= rep  # why?
    out.set_shape(indim)

    return out


@patchmethod(tf.Tensor, tf.Variable)
@op_scope
def repeats(x, repeat_counts, axis=None, name=None):
    """
    element repeat along an axis
    :param x: tensor
    :param repeat_counts: list(int)
    :param axis: list(int)
    :return: same ndim, but size changed
    """
    r = x
    if axis is None:
        axis = range(len(repeat_counts))
    for count, ax in zip(repeat_counts, axis):
        if count == 1:
            continue
        r = repeat(r, count, axis=ax)
    return tf.identity(r, name=name)


def select(b, t, f):
    """
    tf.where equivalent but use broadcasting
    :param b:
    :param t:
    :param f:
    :return:
    """
    b = b.to_float()
    value = b * t + (1.-b) * f
    return value


# region lookup by index

@op_scope
def lookup(value, index, name=None):
    """
    value[(1:, ind)] ? batch value lookup.
    cf) different with nn.embedding_lookup
    :param value: ex) usually prob
    :param index: ex) classification target
    :param name:
    :return:
    """
    if value.ndim > 2:
        return lookup_nd(value, index, name=name)
    else:
        return lookup_2d(value, index, name=name)


@op_scope
def lookup_nd(value, index, name=None):
    """ lookup values by index of last axis """
    k = value.dims[-1]
    value2d = tf.reshape(value, [-1, k])

    v = lookup_2d(value2d, flat(index))

    return tf.reshape(v, tf.shape(index), name=name)


@op_scope
def lookup_2d(value, index, name=None):
    # assert value.dims[0] == index.dims[0]
    # assert value.ndim = index.ndim + 1
    assert value.ndim == 2
    if index.ndim == 2:
        # assert index.dims[1] == 1
        index = tf.squeeze(index, 1)

    batch = tf.shape(index)[0]

    i = tf.range(0, batch)

    ind = tf.transpose(tf.stack([i, index]))

    return tf.gather_nd(value, ind, name=name)

# endregion


@patchmethod(tf.Tensor, tf.Variable)
def top_k(x, k=1, sorted=True, axis=None, name=None):
    """
    axis supported version of tf.nn.top_k
    :param x:
    :param k:
    :param sorted:
    :param axis: default None
    :param name:
    :return:
    """
    if x.dims[-1] == 1:
        x = x.squeeze(-1)

    if axis is None:
        return tf.nn.top_k(x, k=k, sorted=sorted)
    else:
        # support axis
        x = shiftdim(x, -axis - 1)
        res = tf.nn.top_k(x, k=k, sorted=sorted)
        v = shiftdim(res[0], axis + 1)
        res = type(res)(v, res[1])

        return res


@tuple_args
def pack(values, axis=0, name=None):
    """
    pack은 dim(axis)이 늘어남 (concat과 다른점)
    :param values:
    :param axis: default 0
    :return:
    """
    # from r0.12 support axis argument
    # if axis == 0:
    #     return tf.pack(values)
    # else:
    #     # shift axis to 0
    #     # pack and reverse shift
    #     p = tf.pack(values)
    #     return shiftdim(p, axis)
    # return tf.pack(values, axis=axis)
    return tf.stack(values, axis=axis, name=name)


@patchmethod(tf.Tensor, tf.Variable)
def unpack(t, axis=0, num=None, name=None):
    return tf.unstack(t, num=num, axis=axis, name=name)


@patchmethod(tf.Tensor, tf.Variable)
def unstack(t, axis=0, num=None, name=None):
    return tf.unstack(t, num=num, axis=axis, name=name)


@patchmethod(tf.Tensor, tf.Variable)
def splitdim(x, axis, count=None, name=None):
    """ if count is None, same to unpack but, not squeezed output """
    axis = _support_negative_axis(x, axis)
    return tf.split(x, count or x.dims[axis], axis=axis, name=name)


@patchmethod(tf.Tensor, tf.Variable)
def split(x, count=None, axis=0, name=None):
    """ if count is None, same to unpack but, not squeezed output """
    axis = _support_negative_axis(x, axis)
    return tf.split(x, count or x.dims[axis], axis=axis, name=name)


@patchmethod(tf.Tensor, tf.Variable)
def sort(x, axis=None, name=None):
    """

    :param x:
    :param axis: default None means -1
    :param name:
    :return: sorted value
    """
    if axis is None:
        k = tf.shape(x)[-1]
    else:
        k = tf.shape(x)[axis]

    res = top_k(x, k=k, axis=axis, sorted=True, name=name)[0]
    res.set_shape(x.shape)
    return res


@patchmethod(tf.Tensor, tf.Variable)
def argsort(x, axis=None, name=None):
    """

    :param x:
    :param axis: default None means -1
    :param name:
    :return: index
    """
    if axis is None:
        k = tf.shape(x)[-1]
    else:
        k = tf.shape(x)[axis]
    res = top_k(x, k=k, axis=axis, sorted=True, name=name)[1]
    res.set_shape(x.shape)
    return res


@patchmethod(tf.Tensor, tf.Variable)
def one_hot(indices, depth, dtype=const.floatx, axis=-1, name=None, **kwargs):
    """
    # one_hot
    # one_hot = bind(tf.one_hot, on_value=1, off_value=0)
    :param indices:
    :param depth:
    :param dtype:
    :param axis:
    :param kwargs:
    :return:
    """
    onehot = tf.one_hot(indices, depth=depth, dtype=dtype, axis=axis, name=name, **kwargs)
    if indices.dims[axis] == 1:
        if axis >= 0:
            return onehot.squeeze(axis=axis + 1, name=name)
        else:  # axis < 0:
            return onehot.squeeze(axis=axis - 1, name=name)
    return onehot


@patchmethod(tf.Tensor, tf.Variable)
def insert_zero(x, rep, axis=-1, index_after=True, name=None):
    import itertools
    axis = _support_negative_axis(x, axis)
    zeros = tf.zeros_like(x).splitdim(axis)
    if index_after:
        splits = [x.splitdim(axis)] + [zeros for _ in range(rep)]
    else:
        splits = [zeros for _ in range(rep)] + [x.splitdim(axis)]
    ordered = itertools.chain.from_iterable(itertools.izip(*splits))
    # return tf.concat(axis, list(ordered))

    return concat(axis, list(ordered), name=name)


@patchmethod(tf.Tensor, tf.Variable)
def printn(x, first_n=10, message=None, name=None):
    return tf.Print(x, [x], message or x.name, first_n=first_n, name=name)


@patchmethod(tf.Tensor, tf.Variable, name='__rshift__')
def _tensor_right_shift(x, fun):
    # fixme : experimental
    return fun(x)


# region scale util
@patchmethod(tf.Tensor, tf.Variable)
def normalize(x, m, s, name=None):
    return tf.truediv(x - m, s, name=name)


@patchmethod(tf.Tensor, tf.Variable)
def inormalize(x, m, s, name=None):
    """
    inverse normalize x * s + m
    :param x:
    :param m:
    :param s:
    :param name:
    :return:
    """
    return tf.add(x * s, m, name=name)


# endregion

