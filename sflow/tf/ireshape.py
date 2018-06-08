# -*- coding: utf-8 -*-
import sflow.core as tf
from snipy.basic import patchmethod


@patchmethod(tf.Tensor, tf.Variable)
@tf.op_scope
def patch_to_space(x, imgsz, crops=None):
    """
    [batch, patch0, patch1, depth]
    => [batch/blocks, imgsz0, imgsz1, depth]
    == [batch/blocks, block0 x patch0, block1 x patch1, depth]
    :param x:
    :param imgsz:
    :param crops: if None auto crop when need, [(top, bottom), (left, right)]
    :return:
    """
    # if crops is not None:
    #     raise ValueError('not supported yet')

    inshape = x.dims
    batch, patchsz, depth = inshape[0], inshape[1:3], inshape[3]
    bigsz = [imgsz[0] or patchsz[0], imgsz[1] or patchsz[1]]
    blocksz = [bigsz[0] // patchsz[0], bigsz[1] // patchsz[1]]

    if blocksz[0] * patchsz[0] > imgsz[0]:
        blocksz[0] += 1
        bigsz[0] = blocksz[0] * patchsz[0]
    if blocksz[1] * patchsz[1] > imgsz[1]:
        blocksz[1] += 1
        bigsz[1] = blocksz[1] * patchsz[1]

    # # todo consider crops later
    # assert blocksz[0] * patchsz[0] == imgsz[0]
    # assert blocksz[1] * patchsz[1] == imgsz[1]

    # decompose batch
    if batch is not None:
        batch /= (blocksz[0] * blocksz[1])  # new batch size
    data = x.reshape(batch or -1, blocksz[0], blocksz[1], patchsz[0], patchsz[1], depth)

    # transpose to [batch, bloksz[0], patchsz[0], blocksz[1], patchsz[1], depth]
    data = data.transpose(0, 1,3, 2,4, 5)

    # reshape to [batch, imgsz[0], imgsz[1], depth]
    # imgsz = blocksz x patchsz
    data = data.reshape(batch or -1, bigsz[0], bigsz[1], depth)

    # do cropping
    if crops is None:
        if bigsz[0] != imgsz[0] or bigsz[1] != imgsz[1]:
            # data = data[:, :imgsz[0], :imgsz[1], :]
            crops = [bigsz[0] - imgsz[0], bigsz[1] - imgsz[1]]
            crops = [(0,0), (crops[0]//2, crops[0] - crops[0]//2),
                     (crops[1]//2, crops[1] - crops[1]//2), (0, 0)]
            data = tf.crop(data, crops)
    else:
        data = tf.crop(data, crops)

    return data


@patchmethod(tf.Tensor, tf.Variable)
@tf.op_scope
def space_to_patch(x, patchsz, paddings=None):
    """
    [batch, imgsz0, imgsz1, depth]
    => [newbatch, patchsz0, patchsz1, depth]
    => [batch x blocks, imgsz0 / blocksz0, imgsz1 / blocksz1, depth]
    :param tf.Tensor x:
    :param patchsz:
    :param paddings: [(top, bottom), (left, right)] if None then when need, auto paddings
    :return:
    """
    if paddings is not None:
        if len(paddings) == 2:
            paddings = [(0, 0), paddings[0], paddings[1], (0, 0)]
        x = tf.pad(x, paddings)

    inshape = x.dims
    batch, imgsz, depth = inshape[0], inshape[1:3], inshape[3]
    patchsz = [patchsz[0] or imgsz[0], patchsz[1] or imgsz[1]]
    blocksz = (imgsz[0] // patchsz[0], imgsz[1] // patchsz[1])

    if paddings is None:
        # add padding if necessary
        pd = [imgsz[0] - blocksz[0] * patchsz[0], imgsz[1] - blocksz[1] * patchsz[1]]
        if pd[0] or pd[1]:
            pd = [patchsz[0] - pd[0], patchsz[1] - pd[1]]
            paddings = [(0, 0), (pd[0] // 2, pd[0] - pd[0] // 2),
                        (pd[1] // 2, pd[1] - pd[1] // 2), (0, 0)]

            x = tf.pad(x, paddings)
            inshape = x.dims
            batch, imgsz, depth = inshape[0], inshape[1:3], inshape[3]
            blocksz = (imgsz[0] // patchsz[0], imgsz[1] // patchsz[1])

    assert blocksz[0] * patchsz[0] == imgsz[0]
    assert blocksz[1] * patchsz[1] == imgsz[1]

    # batch *= (blocksz[0] * blocksz[1])
    # [batch, imgsz0, imgsz1, d] => [batch, block0, patch0, block1, patch1, d]

    data = x.reshape(batch or -1, blocksz[0], patchsz[0], blocksz[1], patchsz[1], depth)
    # dim to [batch, block0, block1, patch0, patch1, d]
    data = data.transpose(0, 1, 3, 2, 4, 5)
    # new batch size
    if batch is not None:
        batch *= (blocksz[0] * blocksz[1])
    data = data.reshape(batch or -1, patchsz[0], patchsz[1], depth)

    return data


# todo
# build blocks for atrous_conv1d implementation
# time_to_batch
# batch_to_time
# test below functions

@patchmethod(tf.Tensor, tf.Variable)
@tf.op_scope
def time_to_batch(data3d, dilation, paddings=None):
    """
    :param data3d: [batch, time, channel]
    :param dilation: int, rate or dilation
    :param paddings: None or padding formats
    :return: roughly [batch * dilation, time / dilation, channel] with padding if need
    """

    # calc additional pad for depth dim
    batch, width, channel = data3d.dims

    # padding = dilation - (width - (width//dilation) * dilation)
    # if padding != dilation:
    #     # if need, add padding
    #     data3d = tf.pad(data3d, [(0,0), (0, padding), (0, 0)])

    if paddings is not None:
        data3d = tf.pad(data3d, paddings)

    # reshape, dim of time is shrinked along with dilation rate
    data3d = data3d.reshape(-1, dilation, channel)
    data3d = data3d.transpose(1, 0, 2)
    data3d = data3d.reshape(batch * dilation, -1, channel)

    return data3d


@patchmethod(tf.Tensor, tf.Variable)
@tf.op_scope
def batch_to_time(data3d, dilation, crops=None):
    """
    use this after dilation to restore shape to original size
    :param data3d: [batch * dilation, time / dilation, channel]
    :param dilation: int, rate or dilation
    :param crops:
    :return: [batch, time, channel]
    """
    batch, width, channel = data3d.dims

    # batch, width, channel = data3d.dims

    # reshape
    data3d = data3d.reshape(dilation, -1, channel)
    data3d = data3d.transpose(1, 0, 2)
    data3d = data3d.reshape(batch / dilation, -1, channel)

    if crops is not None:
        tops = [crop[0] for crop in crops]
        bottoms = [crop[1] for crop in crops]
        dims = data3d.dims
        size = [dims[0] - tops[0] - bottoms[0],
                dims[1] - tops[1] - bottoms[1],
                dims[2] - tops[2] - bottoms[2]]
        data3d = tf.slice(data3d, tops, size)

    return data3d


@patchmethod(tf.Tensor, tf.Variable)
@tf.op_scope
def channel_to_space(x, r):
    """
    periodic_shuffle implementation
    op for subpixel. subpixl(x, r) := PS(Conv(x, channel*r^2), r)
    periodic shuffle without applying activation to conv output!
    slightly different from depth_to_space in aligning channel
    see: http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf
    or https://arxiv.org/pdf/1609.07009.pdf
    :param x: 4d tensor, [batch, H,W,C], assert mod(C, r^2) == 0
    :param r: int, upscaling factor for one dimension
    :return: [batch, Hr, Wr, c'] tensor
    """
    batch, H, W, C = x.dims
    assert C % (r**2) == 0
    c = C // (r * r)  # result channel

    x = x.reshape(batch, H, W, c, r, r)
    x = x.transpose(0, 1,4, 2,5, 3)
    x = x.reshape(batch, H*r, W*r, c)

    return x


@tf.op_scope
def space_to_channel(x, r):
    """
    inverse op of periodic shuffle
    slightly different from space_to_depth in aliging channel
    :param x: [B, H, W, c]
    :param r: int, space to channel
    :return: [B, H/r, W/r, c*r*r]
    """
    batch, H, W, c = x.dims
    assert H % r == 0 and W % r == 0

    C = c * r * r
    h = H // r
    w = W // r
    x = x.reshape(batch, h, r, w, r, c)
    x = x.transpose(0, 1, 3, 5,2,4)
    x = x.reshape(batch, h, w, C)

    return x


@tf.op_scope
def layer_to_channel(x):
    """
    :param x: [B, Layer, H, W, C]
    :return: [B,H,W, C*Layer]
    """
    assert x.ndim == 5
    batch, L, H, W, c = x.dims
    C = c * L
    # x = x.reshape(batch, H, W, C)
    x = x.transpose(0, 2, 3, 1,4)
    x = x.reshape(batch, H, W, C)
    return x


@tf.op_scope
def channel_to_layer(x, channel):
    """
    :param x: [B, H, W, Layer*C]
    :return: [B, Layer, H, W, C]
    """
    assert x.ndim == 4
    batch, H, W, C = x.dims
    L = C // channel
    # assert
    assert L*channel == C
    x = x.reshape(batch, H, W, L, channel)
    x = x.transpose(0, 3, 1, 2, 4)

    return x


@patchmethod(tf.Tensor, tf.Variable)
@tf.op_scope
def to_space(x):
    """
    call patch_to_space with imgsz auto cacluated
    :param x:
    :return:
    """
    import math
    batch = x.dims[0]
    w = 1
    for h in range(int(math.sqrt(batch)), 1, -1):
        if batch % h == 0:
            w = batch // h
            break

    imgsz = (h * x.dims[1], w * x.dims[2])
    return patch_to_space(x, imgsz)

