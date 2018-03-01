# -*- coding: utf-8 -*-
import sflow.core as tf
import numpy as np

# https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf


def gaussian_kernel(size, sigma=1.5, dtype='float32'):
    """
    gaussian filter for size with sigma
    :param size: int size of kernel
    :param sigma: sigma value default = 1.5
    :param dtype:
    :return: gaussian filter of np.array [h, w, 1, 1]
    """
    # """Function to mimic the 'fspecial' gaussian MATLAB function
    # """
    xi = tf.range(-size//2 + 1, size//2 + 1)
    yi = tf.range(-size//2 + 1, size//2 + 1)
    x, y = tf.meshgrid(xi, yi)
    x = x.astype(dtype)  # indexing='xy' <-default
    y = y.astype(dtype)

    # x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    # x = x.astype(dtype)
    # y = y.astype(dtype)

    v = -((x**2 + y**2)/(2.0*sigma**2))
    # m = np.max(v)
    # old_settings = np.seterr(under='ignore')
    g = tf.exp(v - v.max())
    # np.seterr(**old_settings)

    g /= g.sum()

    # g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    # g /= g.sum()

    return g


# deprecated
# def gaussian_kernel_numpy(size, sigma=1.5, dtype='float32'):
#     """
#     gaussian filter for size with sigma
#     :param size: int size of kernel
#     :param sigma: sigma value default = 1.5
#     :param dtype:
#     :return: gaussian filter of np.array [h, w, 1, 1]
#     """
#     # """Function to mimic the 'fspecial' gaussian MATLAB function
#     # """
#     x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
#
#     x = x.astype(dtype)
#     y = y.astype(dtype)
#
#     v = -((x**2 + y**2)/(2.0*sigma**2))
#     m = np.max(v)
#     old_settings = np.seterr(under='ignore')
#     g = np.exp(v - m)
#     np.seterr(**old_settings)
#
#     g /= g.sum()
#
#     # g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
#     # g /= g.sum()
#
#     return g


def gausssian_kernel2d(size, inchannel, sigma=1.5, dtype='float32'):
    """
    gaussian kerenl tensor for convolution
    :param size:
    :param inchannel: int channel size
    :param sigma:
    :param dtype:
    :return: [size, size, inchannel, 1]
    """
    kernel = gaussian_kernel(size, sigma=sigma, dtype=dtype)
    kernel = tf.expand_dims(kernel, -1)
    kernel = tf.repeat(kernel, rep=inchannel, axis=-1)
    kernel = tf.expand_dims(kernel, -1)

    return kernel

# deprecated
# def gausssian_kernel2d_numpy(size, inchannel, sigma=1.5, dtype='float32'):
#     """
#     gaussian kerenl tensor for convolution
#     :param size:
#     :param inchannel: int channel size
#     :param sigma:
#     :param dtype:
#     :return: [size, size, inchannel, 1]
#     """
#     kernel = gaussian_kernel(size, sigma=sigma, dtype=dtype)
#     kernel = np.expand_dims(kernel, -1)
#     kernel = np.repeat(kernel, repeats=inchannel, axis=-1)
#     kernel = np.expand_dims(kernel, -1)
#     kernel = tf.constant(kernel)
#
#     return kernel


def gaussian_blur(im, window, sigma=1.5, padding='SAME', dtype='float32', kernel=None):
    """

    :param im: [batch, h, w, ch]
    :param window: int size of window
    :param sigma: if increase, more blur
    :param padding: 'same' or 'valid'
    :param dtype: 'float32'
    :param kernel: kernel
    :return: blurred image [batch, h', w', ch] if 'same' same width and height
    """
    if kernel is None:
        inchannel = im.dims[3]
        kernel = gausssian_kernel2d(window, inchannel=inchannel, sigma=sigma, dtype=dtype)

    strides = [1, 1, 1, 1]
    blurred = tf.nn.depthwise_conv2d(im, kernel, strides, padding=padding)

    return blurred


def ssim(x, y, window=11, sigma=1.5, k1=0.01, k2=0.03, l=1., gaussian=True, cs=False):
    """
    ssim similarity [batch,]
    ssimmap similarity between two images
    :param x: img1 [batch, h, w, c]
    :param y: img2 [batch, h, w, c]
    :param window: int, size of local window [window, window]
    :param sigma: sigma of gaussian filter
    :param k1: see paper
    :param k2: see paper
    :param l:
    :param cs: False, return with cs value or not (for mssim)
    :param gaussian: bool, gaussian filter or uniform filter
    :return:
    """
    reduction_indices = (1, 2, 3)
    if cs is False:
        similarity = ssimmap(x, y, window, sigma, k1, k2, l, gaussian, cs=False)
        return similarity.mean(axis=reduction_indices)
    else:
        simmap, csmap = ssimmap(x, y, window, sigma, k1, k2, l, gaussian, cs=True)
        return simmap.mean(axis=reduction_indices), csmap.mean(axis=reduction_indices)


def ms_ssim(x, y, levelweights=None, window=11, sigma=1.5, k1=0.01, k2=0.03, l=1., gaussian=True):
    """
    multiscale ssim
    https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf
    :return: multi scale ssim
    """
    if levelweights is None:
        levelweights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    elif isinstance(levelweights, int):
        orgweights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        levelweights = orgweights[:levelweights]
    level = len(levelweights)
    # weights = tf.constant(levelweights)
    sims = []
    css = 1.0
    for i in range(level):
        s, cs = ssim(x, y, window, sigma, k1, k2, l, gaussian, cs=True)
        sims.append(s)
        css *= cs ** levelweights[i]

        # downsample
        x = tf.nn.avg_pool(x, [1,2,2,1], [1,2,2,1], padding='VALID')
        y = tf.nn.avg_pool(y, [1,2,2,1], [1,2,2,1], padding='VALID')

    mssim = sims[0]**levelweights[0] * css

    return mssim


def ssimmap(x, y, window=11, sigma=1.5, k1=0.01, k2=0.03, l=1., gaussian=True, cs=False):
    """
    ssimmap similarity between two images
    :param x: img1 [batch, h, w, c]
    :param y: img2 [batch, h, w, c]
    :param window: int, size of local window [window, window]
    :param sigma: sigma of gaussian filter
    :param k1: see paper
    :param k2: see paper
    :param l:
    :param gaussian: bool, gaussian filter or uniform filter
    :return: similarity map, shape is equal to shape of conv result with padding='valid'
    """
    c1 = (k1 * l) ** 2
    c2 = (k2 * l) ** 2

    inchannel = x.dims[3]
    if gaussian:
        kernel = gausssian_kernel2d(window, inchannel=inchannel, sigma=sigma)
    else:
        kernel = tf.ones([window, window, inchannel, 1])/(window * window)

    strides = (1, 1, 1, 1)

    # compute (weighted) means
    ux = tf.nn.depthwise_conv2d(x, kernel, strides=strides, padding='VALID')
    uy = tf.nn.depthwise_conv2d(y, kernel, strides=strides, padding='VALID')

    # compute (weighted) variances and covariances
    uxx = tf.nn.depthwise_conv2d(x.square(), kernel, strides=strides, padding='VALID')
    uyy = tf.nn.depthwise_conv2d(y.square(), kernel, strides=strides, padding='VALID')
    uxy = tf.nn.depthwise_conv2d(x * y, kernel, strides=strides, padding='VALID')

    vx = uxx - ux.square()
    vy = uyy - uy.square()
    vxy = uxy - ux*uy

    luminance = (2. * ux * uy + c1) / (ux.square() + uy.square() + c1)
    csvalue = (2. * vxy + c2) / (vx + vy + c2)

    similarity = luminance * csvalue

    if cs is True:
        return similarity, csvalue

    return similarity


def ssim_global(x, y, k1=0.01, k2=0.03, l=1.):
    """
    mmsim : https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf
    wiki : https://en.wikipedia.org/wiki/Structural_similarity
    https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf
    cf. other metrics
    https://pdfs.semanticscholar.org/401e/c2881b20ce7d6dc07c274a1d8f821d4c3841.pdf

    SSIM(x, y) = ((2 µx µy + c1) / (µx^2 + µy^2 + c1)) * ((2 σxy + c2) / (σx^2 + σy^2 + c2))
    todo :
    similarity between images
    considering illumination, constrast and structure of image
    if c1 = c2 = 0, then SSIM == UQI (universal quality index)

    c1 = (k1 * L)^2, c2 = (k2 * L)^2
    k1 = 0.01 and k2 = 0.03 by default
    L = dynamic range of the pixel values (typically 2^(#bits per per pixel - 1)
    c3 - c2 / 2
    :param x: image [batch, h, w, c]
    :param y: image [batch, h, w, c]
    :param c1: todo : check value in original paper
    :param c2: todo : check value in original paper
    :return: similarity
    """
    reduction_indices = (1, 2, 3)
    c1 = (k1 * l) ** 2
    c2 = (k2 * l) ** 2

    mx = x.mean(axis=reduction_indices, keepdims=True)
    my = y.mean(axis=reduction_indices, keepdims=True)
    sx = x.std(axis=reduction_indices, keepdims=True)
    sy = y.std(axis=reduction_indices, keepdims=True)
    xy = (x - mx) * (y - my)
    sxy = xy.mean(axis=reduction_indices, keepdims=True)

    # l(p) : luminace part
    similarity = (2. * mx * my + c1) / (mx.square() + my.square() + c1)
    # cs(p) : contrast x structure (when beta = gamma = 1)
    similarity *= (2 * sxy + c2) / (sx.square() + sy.square() + c2)

    return similarity

