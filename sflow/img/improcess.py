# -*- coding: utf-8 -*-
import sflow.core as tf

from six import wraps
import numpy as np

# https://en.wikipedia.org/wiki/Kernel_(image_processing)


def imfilter(fn):
    @wraps(fn)
    def _wrapped(axis='x'):
        f = tf.constant(fn(), dtype=tf.float32)
        if axis != 'x':
            f = f.T
        f = f.expand_dims(-1).expand_dims(-1, name=fn.__name__)
        return f
    return _wrapped


@imfilter
def sobel_filter():
    # https://en.wikipedia.org/wiki/Sobel_operator
    return [[-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]]


@imfilter
def scharr_filter():
    return [[-3, 0, 3],
            [-10, 0, 10],
            [-3., 0, 3]]


@imfilter
def laplacian_filter():
    return [[0, 1, 0],
            [1, -4, 0],
            [0, 1, 0]]


@imfilter
def laplacian_diagonal_filter():
    return [[1, 1, 1],
            [1, -8, 1],
            [1, 1, 1]]


@imfilter
def sharpen_filter():
    return [[-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]]

# todo make flexible size with gaussian filter
# @imfilter
# def unsharp_filter():
#     import numpy as np
#     a = [[1, 4, 6, 4, 1],
#          [4, 16, 24, 16, 4],
#          [6, 24, -476, 24, 6],
#          [4, 16, 24, 16, 4],
#          [1, 4, 6, 4, 1]]
#     return -1/256. * np.asarray(a, dtype=np.float)


# @imfilter
# def boxblur_filter():
#     return [[1/9., 1/9., 1/9.],
#             [1/9., 1/9., 1/9.],
#             [1/9., 1/9., 1/9.]]


def axis_dwconv(x, ifilter, stride, padding, name=None):
    f = ifilter.repeat(x.shapes[-1], axis=2)
    return tf.nn.depthwise_conv2d(x, f, strides=stride, padding=padding, name=name)


def _standardize_stride(stride):
    """
    standardize_stride
    :param stride:
    :return:
    """
    if isinstance(stride, int):
        stride = (stride, stride)
    return 1, stride[0], stride[1], 1


def _standarize_filter(f, name=None):
    f = tf.convert_to_tensor(f, dtype=tf.float32)
    if f.ndim == 2:
        f = f.expand_dims(-1).expand_dims(-1, name=name)
    return f


def sharpen(x, filter=None, stride=1, padding='SAME', name=None):
    if filter is None:
        filter = sharpen_filter()
    else:
        filter = _standarize_filter(filter)

    return axis_dwconv(x, filter, _standardize_stride(stride),
                       padding, name=name or 'sharpen')


def edge_enhance(x, stride=1, padding='SAME', name=None):
    # https://www.packtpub.com/mapt/book/Application+Development/9781785283932/2/ch02lvl1sec22/Sharpening
    kernel_sharpen_1 = _standarize_filter([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    kernel_sharpen_2 = _standarize_filter([[1, 1, 1], [1, -7, 1], [1,1, 1]])
    kernel_sharpen_3 = np.array([[-1, -1, -1, -1, -1],
                                 [-1, 2, 2, 2, -1],
                                 [-1, 2, 8, 2, -1],
                                 [-1, 2, 2, 2, -1],
                                 [-1, -1, -1, -1, -1]]) / 8.0
    stride = _standardize_stride(stride)
    kernel_sharpen_3 = _standarize_filter(kernel_sharpen_3)

    x = axis_dwconv(x, kernel_sharpen_1, stride, padding)
    x = axis_dwconv(x, kernel_sharpen_2, stride, padding)
    x = axis_dwconv(x, kernel_sharpen_3, stride, padding, name=(name or 'edge_enhance'))

    return x


# filters http://docs.opencv.org/3.0-beta/modules/imgproc/doc/filtering.html?highlight=scharr#cv2.Scharr

# region sobel

def sobelx(x, stride=1, padding='SAME', name=None):
    return axis_dwconv(x, sobel_filter('x'),
                       _standardize_stride(stride), padding,
                       name=name or 'sobelx')


def sobely(x, stride=1, padding='SAME', name=None):
    return axis_dwconv(x, sobel_filter('y'),
                       _standardize_stride(stride), padding,
                       name=name or 'sobely')


def sobelxy(x, stride, padding):
    # https://en.wikipedia.org/wiki/Sobel_operator
    assert x.ndim == 4
    gx = sobelx(x, stride, padding)
    gy = sobely(x, stride, padding)

    return gx, gy


def sobel(x, stride=1, padding='SAME', name=None):
    # apply sobel filter

    def sobel2d(x):
        gx, gy = sobelxy(x, stride, padding)
        g = tf.sqrt(gx.square() + gy.square(), name=name or 'sobel2d')
        return g

    assert x.ndim == 4
    # todo 3d

    return sobel2d(x)

# endregion

# region scharr


def scharrx(x, stride=1, padding='SAME', name=None):
    return axis_dwconv(x, scharr_filter('x'),
                       _standardize_stride(stride), padding,
                       name=name or 'scharrx')


def scharry(x, stride=1, padding='SAME', name=None):
    return axis_dwconv(x, scharr_filter('y'),
                       _standardize_stride(stride), padding,
                       name=name or 'scharry')


def scharrxy(x, stride, padding):
    # https://en.wikipedia.org/wiki/Sobel_operator
    assert x.ndim == 4
    gx = scharrx(x, stride, padding)
    gy = scharry(x, stride, padding)

    return gx, gy


def scharr(x, stride=1, padding='SAME', name=None):
    # apply sobel filter

    def sobel2d(x):
        gx, gy = scharrxy(x, stride, padding)
        g = tf.sqrt(gx.square() + gy.square(), name=name or 'scharr2d')
        return g

    assert x.ndim == 4
    # todo 3d

    return sobel2d(x)

# endregion


# region gradient


def sobel_grad_angle(x, stride=1, padding='SAME', eps=0.):
    # pass
    gx, gy = sobelxy(x, stride, padding)
    return tf.tanh(gx/(gy+eps))


def scharr_grad_angle(x, stride=1, padding='SAME', eps=0.):
    gx, gy = scharrxy(x, stride, padding)
    return tf.tanh(gx/(gy+eps))


def laplacian(x, diagonal=True, name=None):

    kernel = laplacian_diagonal_filter() if diagonal else laplacian_filter()
    res = axis_dwconv(x, kernel, _standardize_stride(1), padding='SAME',
                      name=(name or 'laplacian'))
    return res


# endregion


def get_affine_transform(src, dst, name=None):
    """
    src
    :param src: [n, 2]
    :param dst: [n, 2]
    :param name: output name
    :return: [2, 3] affine matrix
    """
    n = src.dims[0]
    src = tf.concat(0, [src.T, tf.ones(n, dtype=tf.float32)])
    out = tf.matrix_solve_ls(src, tf.eye(3, dtype=tf.float32))
    aff = tf.matmul(src, out, name=name or 'affine_matrix')

    return aff


def warp_affine(src, transform):
    import tensorflow as tf
    # transform
    # tf.contrib.image.transform(src, transform)
    # pass
    # todo do..
    pass


# test implementation
def sketch(x, threshold=None, digitize=False, invert=True, gaussian=True, gwindows=11,
           gsigma=0.5, keep_alpha=True, keep_channel=False, gray_to_rgb=None):
    """
    # test implementation
    one sobel filter
    convert images to sketch-like images

    :param x: image [BHWC]
    :param invert: make edge to black (inverted) or not
    :param threshold: None for output asis else remove under threshhold value
    :param digitize: 0 or 1
    :param gaussian: use gaussian or not
    :param gwindows: gaussian filter window size
    :param gsigma: gaussian filter sigma
    :param keep_alpha:
    :param keep_channel: False
    :param gray_to_rgb: None, True, False
    :return: image
    """
    # blur
    # color to gray
    # sobel filter
    # inverse image
    from ssim import gaussian_blur

    assert x.ndim == 4

    keep_alpha = keep_alpha and x.dims[-1] == 4
    if keep_alpha:
        alpha = x[:, :, :, 3:]

    if gaussian:
        x = gaussian_blur(x, gwindows, gsigma)

    if keep_channel:
        x = x[:, :, :, :3]
    elif x.dims[-1] == 4:
        # 2channel output : gray, alpha channel
        x = tf.image.rgb_to_grayscale(x[:, :, :, :3])
        # todo add alpha part
    elif x.dims[-1] == 3:
        x = tf.image.rgb_to_grayscale(x)

    x = sobel(x) / 4.  # normalize
    if not keep_channel:
        x = x.mean(axis=3, keepdims=True)
    if digitize:
        x = tf.cast(x >= (threshold or 0.5), dtype=x.dtype)
    if threshold is not None:
        x = x * tf.cast(x >= threshold, dtype=x.dtype)

    if invert:
        x = 1.0 - x
    if gray_to_rgb and x.dims[-1] == 1:
        x = tf.image.grayscale_to_rgb(x)
    if keep_alpha:
        x = x.cat(alpha)

    return x


