# -*- coding: utf-8 -*-
from __future__ import absolute_import
import sflow.core as tf


def coord2d(shape, center=None):
    """
    :param shape:
    :param center:
    :return: x[1,W], y[H,1] vector, to use broadcasting
    """
    if center:
        raise NotImplementedError

    # x = tf.ones(shape=shape, dtype=tf.float32).cumsum(axis=0) / shape[0]
    # y = tf.ones(shape=shape, dtype=tf.float32).cumsum(axis=1) / shape[1]

    x = tf.linspace(0., 1., shape[1]).expand_dims(0).to_float()
    y = tf.linspace(0., 1., shape[0]).expand_dims(1).to_float()

    return x, y


def rand_occlusion(shape, count, starts=None, sizes=None, dtype=tf.bool):
    """

    :param shape: [B,H,W]
    :param count: int or range [min, max]
    :param starts: [(xmin, xmax), (ymin, ymax)]
    :param sizes: [(xmin, xmax), (ymin, ymax)]
    :return: [B,H,W] bool
    """
    # todo refactor parameters

    if not isinstance(count, int):
        count = tf.random_uniform(shape=(), minval=count[0], maxval=count[1])

    bbox = rand_bbox(shape, count, starts=starts, sizes=sizes, dtype=dtype)
    mask = bbox3_to_mask(bbox, shape[1:3])  # B,Count,H,W
    mask = mask.any(axis=1)

    return mask


def rand_cbox(shape, count, starts=None, sizes=None):
    """

    :param shape:
    :param count:
    :param starts:
    :param sizes:
    :return: random colored box
    """
    from .blend import composite
    bbox = rand_bbox(shape, count, starts=starts, sizes=sizes)
    mask = bbox3_to_mask(bbox, shape[1:3]).expand_dims(-1)  # B,Count,H,W, 1
    cshape = [bbox.dims[0] or tf.shape(bbox)[0], count, 1, 1, 3]
    colors = tf.random_uniform(shape=cshape)

    colorbox = tf.select(mask, colors, 0.)
    colorbox = composite(colorbox, order='BL')
    return colorbox


def rand_bbox(shape, count, starts=None, sizes=None, dtype=tf.bool):
    if starts is None:
        starts = [(0.0, 1.0), (0.0, 1.0)]
    if sizes is None:
        sizes = [(0.0, 1.0), (0.0, 1.0)]

    rand_shape = tf.stack([shape[0], count, 1], axis=0).to_int32()
    xstart = tf.random_uniform(shape=rand_shape, minval=starts[0][0], maxval=starts[0][1])
    ystart = tf.random_uniform(shape=rand_shape, minval=starts[1][0], maxval=starts[1][1])
    xsize = tf.random_uniform(shape=rand_shape, minval=sizes[0][0], maxval=sizes[0][1])
    ysize = tf.random_uniform(shape=rand_shape, minval=sizes[1][0], maxval=sizes[1][1])

    bbox = tf.concat(-1, [xstart, ystart, xsize, ysize])
    bbox.set_shape([shape[0], None, 4])

    return bbox


def bbox_to_mask(bbox, shape):
    # assert shape.ndim == 2
    ndim = bbox.ndim
    if ndim == 1:
        return bbox1_to_mask(bbox, shape)
    elif ndim == 2:
        return bbox2_to_mask(bbox, shape)
    elif ndim == 3:
        return bbox3_to_mask(bbox, shape)
    else:
        raise ValueError


def bbox3_to_mask(bbox3, shape):
    """
    :param bbox3: [B, N, 4]  y_from,x_from, y_size, x_size, normalized 0~1
    :shape [H,W]
    :return: [B, N, H, W] mask
    """
    assert bbox3.ndim == 3 and bbox3.dims[-1] == 4

    x, y = coord2d(shape)  # [H, W]
    x = x.expand_dims(0).expand_dims(0)   # 1,1,H,W
    y = y.expand_dims(0).expand_dims(0)   # 1,1,H,W

    bbox3 = bbox3.expand_dims(-1).expand_dims(-1)
    xfrom = bbox3[:, :, 1]  # B,N,1,1
    xto = xfrom + bbox3[:, :, 3]

    yfrom = bbox3[:, :, 0]  # B,N,1,1
    yto = yfrom + bbox3[:, :, 2]

    # broadcasting
    ix = tf.logical_and(x >= xfrom, x <= xto)  # B,N,H,W
    iy = tf.logical_and(y >= yfrom, y <= yto)  # B,N,H,W

    imask = tf.logical_and(ix, iy)

    return imask


# def bbox_mask_any(bbox2, shape):
#     return bbox2_to_mask(bbox2, shape).any(axis=0)


def bbox2_to_mask(bbox2, shape):
    """

    :param bbox2: [N, 4] y_from,x_from, y_size, x_size, normalized 0~1
    :param shape: (H, W)
    :return: [N, H, W] mask
    """

    # same to this
    # return tf.map_fn(lambda x: bbox1_to_mask(x, shape), bbox3, dtype=tf.bool)
    assert bbox2.ndim == 2 and bbox2.dims[-1] == 4

    x, y = coord2d(shape)  # H, W
    x = x.expand_dims(0)   # 1HW
    y = y.expand_dims(0)   # 1HW

    bbox2 = bbox2.expand_dims(-1).expand_dims(-1)
    xfrom = bbox2[:, 1]  # N,1,1
    xto = xfrom + bbox2[:, 3]

    yfrom = bbox2[:, 0]  # N,1,1
    yto = yfrom + bbox2[:, 2]

    ix = tf.logical_and(x >= xfrom, x <= xto)
    iy = tf.logical_and(y >= yfrom, y <= yto)

    imask = tf.logical_and(ix, iy)

    return imask


def bbox1_to_mask(bbox1, shape):
    """
    bbox
    :param bbox1: [4] float32 value of (0~1) y_from,x_from, y_size, x_size, normalized 0~1
    :param shape: [H,W]
    :return: mask of dtype tf.bool
    """

    # ex: shape = (255, 255)
    # bbox = (N, 4)
    assert bbox1.ndim == 1 and bbox1.dims[-1] == 4
    x, y = coord2d(shape)

    # check bbox x, bbox[1]~bbox[1]+bbox[3]
    # ix = bbox[1] <= x <= (bbox[1] + bbox[3])
    # iy = bbox[0] <= y <= (bbox[0] + bbox[2])
    ix = tf.logical_and(x >= bbox1[1], x <= (bbox1[1] + bbox1[3]))
    iy = tf.logical_and(y >= bbox1[0], y <= (bbox1[0] + bbox1[2]))

    imask = tf.logical_and(ix, iy)

    return imask


def setitem(x, slices, values):
    # todo : move to ipatch.. use mask and where? select function
    # tensor.value??

    raise NotImplementedError


if __name__ == '__main__':
    import sflow.py as py

    bbox = tf.constant([0.2, 0.4, 0.5, 0.2])
    shape = (255, 255)
    out = bbox2_to_mask(bbox, shape)
    o = out.eval()

    py.plt.matshow(o, cmap='gray')
    py.plt.plot_pause()
