# -*- coding: utf-8 -*-
from __future__ import absolute_import
import sflow.core as tf
from six import wraps
from tensorflow.python.ops import control_flow_ops

# region helpers


def _standize_input_tuple(*args):
    if len(args) == 1:
        if isinstance(args[0], tuple):
            return args[0]
        elif isinstance(args[0], list):
            return tuple(args[0])
    return args


def _standize_output_res(res):
    if isinstance(res, (tuple, list)) and len(res) == 1:
        return res[0]
    else:
        return res


def _standize_output_fun_or_res(fun, *args):
    @wraps(fun)
    def wrap(*args_):
        args_ = _standize_input_tuple(*args_)
        res = fun(*args_)
        return _standize_output_res(res)

    if len(args) == 0:
        return wrap
    else:
        return wrap(*args)


def _rand_apply_batch_fun(fun, imagez, p=0.5, **kwargs):
    """
    :param fun: fun
    :param imagez: tuple(image)
    :param p: prob of applying fun
    :return:
    """

    assert isinstance(imagez, (tuple, list))
    # assert all shape equals to each other
    # assume shape(x) == shape(fun(x))
    shape = (imagez[0].dims[0],)
    irand = tf.random_choice(shape, p=p)

    applied = tuple(fun(im, **kwargs) for im in imagez)
    outs = tuple(tf.where(irand, im, ap) for im, ap in zip(imagez, applied))

    return _standize_output_res(outs)


def _rand_apply_fun(fun, imagez, p=0.5):
    assert isinstance(imagez, (tuple, list))

    shape = (imagez[0].dims[0],)
    irand = tf.random_choice(shape, p=p)
    applied = [tf.map_fn(fun, images) for images in imagez]
    outs = tuple(tf.where(irand, images, ap) for images, ap in zip(imagez, applied))

    return _standize_output_res(outs)


# endregion

# region flip

@tf.op_scope
def flipud(*images):
    """
    flipud for 4d image
    :param images:
    :return:
    """
    def _flipud(img):
        return tf.map_fn(tf.image.flip_up_down, img)

    return _standize_output_res(list(map(_flipud, images)))


def rand_flipud(*imagez, **kwargs):
    """
    example::
        ex1)
        f = rand_flipud(p=0.5)
        f(images)  # or f(images, images2)
        ex2)
        res = rand_flipud(images, images2, p=0.5)
        res = rand_flipud([images, images2], p=0.5)

    flip together
    apply rand_fliplr to pairs
    :param imagez:
    :param p: 0.5
    :return: fun or res
    """
    p = kwargs.pop('p', 0.5)

    @tf.op_scope('rand_flipud')
    def _rand_flipud(*args):
        return _rand_apply_batch_fun(flipud, args, p=p)

    return _standize_output_fun_or_res(_rand_flipud, *imagez)


def fliplr(*images):
    """
    4d image flip left right
    :param images:
    :return:
    """
    @tf.op_scope('fliplr')
    def _fliplr(im):
        return tf.map_fn(tf.image.flip_left_right, im)

    return _standize_output_res(list(map(_fliplr, images)))


def rand_fliplr(*imagez, **kwargs):
    # """
    # flip together
    # apply rand_fliplr to pairs
    # :param imagez:
    # :param p: 0.5
    # :return:
    # """
    # # return _rand_apply_image_fun(tf.image.flip_left_right, imagez, **kwargs)
    # return _rand_apply_batch_fun(fliplr, imagez, **kwargs)
    """
    example::
        ex1)
        f = rand_flipud(p=0.5)
        f(images)  # or f(images, images2)
        ex2)
        res = rand_flipud(images, images2, p=0.5)
        res = rand_flipud([images, images2], p=0.5)

    flip together
    apply rand_fliplr to pairs
    :param imagez:
    :param p: 0.5
    :return: fun or res
    """
    p = kwargs.pop('p', 0.5)

    @tf.op_scope('rand_fliplr')
    def _rand_fliplr(*args):
        return _rand_apply_batch_fun(fliplr, args, p=p)

    return _standize_output_fun_or_res(_rand_fliplr, *imagez)

# endregion

# region transform


def transform(transforms, *images):

    @wraps(tf.contrib.image.transform)
    def _transform(*imgs):
        return tuple(tf.contrib.image.transform(im, transforms) for im in imgs)

    return _standize_output_fun_or_res(_transform, *images)


# endregion

# region transpose

def transpose_img(*images):

    @tf.op_scope('transpose_img')
    def _transpose_img(img4d):
        return tf.map_fn(tf.image.transpose_image, img4d)

    return _standize_output_res(list(map(_transpose_img, images)))


@tf.op_scope
def rand_transpose_img(*imagez, **kwargs):
    return _rand_apply_batch_fun(rand_fliplr, imagez, **kwargs)

# endregion

# region rotate


def rot90(*imagez, **kwargs):
    """
    4d image rot90
    :param imagez:
    :return: fun or res
    """
    k = kwargs.pop('k', 1)

    def _rot90_3r(x, k_):
        out = tf.image.rot90(x, k_)
        dims = x.dims
        if k_ == 1 or k_ == 3:
            out.set_shape([dims[1], dims[0]] + dims[2:])
        if k_ == 2:
            out.set_shape(dims)
        return out

    @tf.op_scope('rot90')
    def _rot90(*imgs):
        for im in imgs:
            tf.assert_rank(im, 4)
        return tuple(tf.map_fn(lambda x: _rot90_3r(x, k), im) for im in imgs)

    return _standize_output_fun_or_res(_rot90, *imagez)


def rand_rot90(*imagez, **kwargs):
    k = kwargs.pop('k', 1)
    p = kwargs.pop('p', 0.5)

    @tf.op_scope('rand_rot90')
    def _rand_rot90(*args):
        return _rand_apply_batch_fun(rot90, args, p=p, k=k)

    return _standize_output_fun_or_res(_rand_rot90, *imagez)

# todo rotate for 3d tensor


def rotates(angles, *images, **kwargs):
    """
    example::

        a = tf.ones((10, 20, 30, 3)).cumsum(axis=1).cumsum(axis=0)/400.
        a2 = tf.ones((10, 20, 30, 3)).cumsum(axis=1).cumsum(axis=0)/400.
        # c1, c2 = tf.img.rotate(tf.pi*0.25, a, a2)
        # f = tf.img.rand_crop((5, 5))
        f = tf.img.rotate(tf.pi*0.25)
        c1, c2 = f(a, a2)

    see tf.contrib.image.rotate
    :param angles: angles
    :param images: images
    :return: fun or res
    """
    def _rotate(*imgs):
        return tuple(rotate(im, angles, **kwargs) for im in imgs)

    return _standize_output_fun_or_res(_rotate, *images)


@tf.op_scope
def rotate(img, angles, outsize=None, oob=None):
    """
    rotate image or images by angle or angles
    :param img: [bhwc] or [hwc]
    :param angles: radian angle tensor
    :param outsize: output shape or None for same shape to img
    :param oob: color of out of bound
    :return:
    """
    from . import transforms as t  # angle_to_theta, transform)
    # angles =

    # todo : check angle is radian or not

    theta = t.angle_to_theta(-angles)
    return t.transform(img, theta, outsize=outsize, oob=oob)


def rand_rotate(*imagez, **kwargs):
    """
    :param imagez: 4d
    :param angles: [minval, maxval] of radian angle
    :return:
    """
    angles = kwargs.pop('angles',  (-tf.pi, tf.pi))

    @tf.op_scope('rand_rotate')
    def _rand_rotate(*imgz):

        batch = (imgz[0].dims[0],)
        angrange = tf.random_uniform(batch, angles[0], angles[1])

        return rotates(angrange, *imgz)

    return _standize_output_fun_or_res(_rand_rotate, *imagez)

# endregion

# region crop


def rand_crop(sz, *imagez):
    """
    example::

        a = tf.ones((10, 20, 20, 3)).cumsum(axis=1).cumsum(axis=0)/400.
        a2 = tf.ones((10, 20, 20, 3)).cumsum(axis=1).cumsum(axis=0)/400.
        # c1, c2 = tf.img.rand_crop((10, 10), a, a2)
        f = tf.img.rand_crop((10, 10))
        c1, c2 = f(a, a2)
        print tf.all(c1.equal(c2)).eval()

    :param sz:
    :param imagez:
    :return: func or res
    """

    @tf.op_scope('rand_crop')
    def _rand_crop(*imgs):
        with tf.name_scope(None, 'rand_crop', list(imgs) + [sz]):
            value = imgs[0]
            size = tf.convert_to_tensor(sz, dtype=tf.int32, name="size")
            shape = tf.shape(value)[1:3]  # HW of BHWC

            check = tf.Assert(tf.reduce_all(shape >= size), ["Need value.shape >= size, got", shape, size])
            shape = control_flow_ops.with_dependencies([check], shape)

            # assert same shape
            for v in imgs:
                vshape = tf.shape(v)[1:3]  # assert v.ndim == 4
                check = tf.Assert(tf.reduce_all(shape.equal(vshape)),
                                  ["Need same (H,W,?) image.shape[1:3] == otherimage.shape[1:3], got", shape, vshape])
                shape = control_flow_ops.with_dependencies([check], shape)

            limit = shape - size + 1
            batchshape = tf.shape(value)[:1].append(2)
            # offsets = tf.random_uniform(batchshape, dtype=size.dtype, maxval=size.dtype.max) % limit  # add seed
            offsets = tf.random_uniform(batchshape, maxval=limit, dtype=tf.int32)

            # sz = size
            size = size.append(-1)

            def _3d_crop(values, offset):
                # values, offset = args
                offset = offset.append(0)
                # outs = [tf.slice(img, offset, size) for img in values]
                outs = []
                for img in values:
                    out = tf.slice(img, offset, size)
                    out.set_shape(list(sz)+v.dims[-1:])
                    outs.append(out)
                return outs

            return tf.map_fn(_3d_crop, [imgs, offsets], dtype=[v.dtype for v in imgs])

    return _standize_output_fun_or_res(_rand_crop, *imagez)


def rand_crop3d(sz, *images, **kwargs):
    """
    assume imagez has same sz (H, W, ?)
    size of iamgez at least sz, cf) tf.random_crop
    example::

        c1, c2 = rand_crop3d((10,10), tf.ones((20,20,4)), tf.ones((20,20,6)))
        In [1]: c1
        Out[1]: <tf.Tensor 'rand_crop_1/Slice:0' shape=(10, 10, 4) dtype=float32>

        In [2]: c2
        Out[2]: <tf.Tensor 'rand_crop_1/Slice_1:0' shape=(10, 10, 6) dtype=float32>

    :param sz: target size
    :param images:
    :return:
    """
    # modified from source of tf.random_crop

    @tf.op_scope('rand_crop3d')
    def _rand_crop3d(*imgs):

        with tf.name_scope(kwargs.pop('name', None), 'rand_crop', list(imgs) + [sz]):
            value = imgs[0]
            size = tf.convert_to_tensor(sz, dtype=tf.int32, name="size")
            shape = tf.shape(value)[:2]

            check = tf.Assert(tf.reduce_all(shape >= size), ["Need value.shape >= size, got", shape, size])
            shape = control_flow_ops.with_dependencies([check], shape)

            # assert same shape
            for v in imgs:
                vshape = tf.shape(v)[:2]
                check = tf.Assert(tf.reduce_all(shape.equal(vshape)),
                                  ["Need same (H,W,?) image.shape[:2] == otherimage.shape[:2], got", shape, vshape])
                shape = control_flow_ops.with_dependencies([check], shape)

            limit = shape - size + 1
            offset = tf.random_uniform(tf.shape(shape), dtype=size.dtype, maxval=size.dtype.max) % limit  # add seed
            # take last dim as-is
            # tf.assert_greater_equal(offset, 0)
            # tf.assert_greater_equal(size, 0)
            offset = offset.append(0)
            size = size.append(-1)

            return tuple(tf.slice(v, offset, size) for v in imgs)

    return _standize_output_fun_or_res(_rand_crop3d, *images)


def rand_crop_offsets(sz, *imagez):
    """
    return with offset random value
    example::

        a = tf.ones((10, 20, 20, 3)).cumsum(axis=1).cumsum(axis=0)/400.
        a2 = tf.ones((10, 20, 20, 3)).cumsum(axis=1).cumsum(axis=0)/400.
        # c1, c2 = tf.img.rand_crop((10, 10), a, a2)
        f, offsets = tf.img.rand_crop_offsets((10, 10))
        c1, c2 = f(a, a2)
        print tf.all(c1.equal(c2)).eval()

    :param sz:
    :param imagez:
    :return: func or res
    """

    def _rand_crop_offsets(*imgs):
        with tf.name_scope(None, 'rand_crop', list(imgs) + [sz]):
            value = imgs[0]
            size = tf.convert_to_tensor(sz, dtype=tf.int32, name="size")
            shape = tf.shape(value)[1:3]  # HW of BHWC

            check = tf.Assert(tf.reduce_all(shape >= size), ["Need value.shape >= size, got", shape, size])
            shape = control_flow_ops.with_dependencies([check], shape)

            # assert same shape
            for v in imgs:
                vshape = tf.shape(v)[1:3]  # assert v.ndim == 4
                check = tf.Assert(tf.reduce_all(shape.equal(vshape)),
                                  ["Need same (H,W,?) image.shape[1:3] == otherimage.shape[1:3], got", shape, vshape])
                shape = control_flow_ops.with_dependencies([check], shape)

            limit = shape - size + 1
            if value.dims[0] is None:
                batchshape = tf.shape(value)[:1].append(2)
            else:
                batchshape = (value.dims[0], 2)

            offsets = tf.random_uniform(batchshape, dtype=size.dtype, maxval=size.dtype.max) % limit  # add seed
            # offsets = tf.random_uniform(batchshape, maxval=limit, dtype=tf.int32)

            # sz = size
            size = size.append(-1)

            def _3d_crop(args):
                values, offset = args
                offset = offset.append(0)
                # outs = [tf.slice(img, offset, size) for img in values]
                outs = []
                for img in values:
                    out = tf.slice(img, offset, size)
                    out.set_shape(list(sz)+v.dims[-1:])
                    outs.append(out)
                return outs

            return tf.map_fn(_3d_crop, [imgs, offsets], dtype=[v.dtype for v in imgs]), offsets

    return _rand_crop_offsets(*imagez)


def crop_center(sz, *images):

    def _crop_center_one(imgs, name=None):
        size = tf.convert_to_tensor(sz, dtype=tf.int32, name="size")
        hw = tf.shape(imgs)[-3:-1]

        # no gpu support
        # check = tf.Assert(tf.reduce_all(hw >= size),
        #                   ['Need crop size less than tensor tensor.shape[-3:-1] >= cropsize, got', hw, size])
        # hw = control_flow_ops.with_dependencies([check], hw)

        offset = (hw - size) // 2
        if imgs.ndim == 3:
            offset = tf.concat(0, [offset, [0]])
            size = tf.concat(0, [size, [-1]])
        if imgs.ndim == 4:
            offset = tf.concat(0, [[0], offset, [0]])
            size = tf.concat(0, [[-1], size, [-1]])

        return tf.slice(imgs, offset, size, name=name)

    @tf.op_scope('crop_center')
    def _crop_center(*imgz, **_):
        return tuple(_crop_center_one(img) for img in imgz)

    return _standize_output_fun_or_res(_crop_center, *images)

# endregion

# region pad


def pad_if_need(image, size, offsets=None):
    """
    :param image: tensor3d[H,W,C]
    :param size: (int, int) targetsize (H,W)
    :param offsets: (0,0) for None
    :return:
    """
    assert image.ndim == 3
    imshape = tf.shape(image)

    # get target shape if possible
    tshape = image.dims
    for i in (0, 1):
        if tshape[i] is not None and size[i] > tshape[i]:
            tshape[i] = size[i]

    targetshape = tf.convert_to_tensor(size).append(imshape[-1])
    need = targetshape - imshape
    # padding need
    need = tf.where(need > 0, need, tf.zeros(tf.shape(need), dtype=tf.int32))
    if offsets is None:
        offsets = [0, 0, 0]
    else:
        offsets = list(offsets)
        offsets.append(0)

    # upper padding = need // 2

    padding_first = need // 2 + tf.convert_to_tensor(offsets)
    padding_left = need - padding_first
    padding = tf.concat(0, [[padding_first], [padding_left]]).T

    out = tf.pad(image, padding, 'CONSTANT')
    # rshape = tf.maximum(imshape, targetshape)

    # if known shape.. set
    out.set_shape(tshape)

    return out

# endregion

# region simple imread to tensor


def imread(fpath, size=None, dtype='float32'):
    # read image as tensor
    import sflow.python.fileutil as py
    img = py.imread(fpath, size=size, expand=True, dtype=dtype)

    # from skimage import io, transform
    # img = io.imread(fpath)
    # if size is not None:
    #     sz = list(img.shape)
    #     sz[:len(size)] = size
    #     img = transform.resize(img, sz, preserve_range=True)
    # img = img.astype('float32') / 255.
    # img = tf.convert_to_tensor(img)
    # img = img.expand_dims(0)
    # if img.ndim == 3:
    #     img = img.expand_dims(-1)
    # return img

    return tf.convert_to_tensor(img)


# endregion


# region jittering

# def jitter(img, shifts, axis=(1, 2), padmode='CONSTANT'):
#     """
#
#     :param img: [BHWC]
#     :param shift: [H,W] or [BHW]
#     :return: jittered image
#     """
#     # todo jitter image
#     pass
#
#
# def rand_jitter(imgz):
#     pass


# endregion


if __name__ == '__main__':
    from sflow.sample import coffee
    import sflow.py as py

    img = coffee(True)
    a = rotate(img, tf.pi)
    py.imshow([img, a])
    py.plot_pause()
