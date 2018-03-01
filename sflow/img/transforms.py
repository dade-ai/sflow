# -*- coding: utf-8 -*-
import sflow.core as tf

# todo : fixme no batch version.. for random tranforms
# N to 1 , N to N

# region transform


def angle_to_theta(angles):
    """
    algle to 2x3 transform matrix theta
    :param angles: assume radian
    :return: outdim : [angleshapes, 2, 3]
    """
    c = tf.cos(angles)
    s = tf.sin(angles)
    z = tf.zeros_like(angles)
    t = tf.stack([[c, -s, z],
                  [s, c, z]])
    t = t.shiftdim(-2)
    return t


@tf.op_scope
def transform(img, theta, outsize=None, oob=None):
    """
    sampling from source coord by theta (xy_src = theta * xy_target !!!)
    :param img: [BHWC] or [HWC]
    :param theta: [2x3] or [B, 2x3] if img [BHWC] inverse theta!!!
    :param outsize: [H,W] or None if none same HW to img
    :param oob: outofbound color [C,]
    :return:
    """

    # tf.assert_type(img, tf_type=tf.float32)

    if theta.ndim == 2:
        # [2, 3]
        # assert theta.shape == (2,3)
        if img.ndim == 4:
            # outdim : [BHWC]
            return transform_4r(img, theta, outsize=outsize, oob=oob)
        elif img.ndim == 3:
            # outdim : [HWC]
            return transform_3r(img, theta, outsize=outsize, oob=oob)
        else:
            raise ValueError('2d support in transform?')
    elif theta.ndim == 3:
        # theta [B,2,3]
        # assert theta.shapes[1:3] == (2,3)
        if img.ndim == 4:
            if img.dims[0] is not None and theta.dims[0] is not None:
                assert img.dims[0] == theta.dims[0]
            else:
                tf.assert_equal(img.shape[0], theta.shape[0])

            # assert img.shapes[0] == theta.shapes[0], or tf.Assert()...
            # outdim : [BHWC]
            return tf.map_fn(lambda x: transform_3r(x[0], x[1], outsize=outsize, oob=oob),
                             [img, theta], dtype=tf.float32)
        elif img.ndim == 3:
            # one image to multi transform???
            # [BHWC] !
            return tf.map_fn(lambda t: transform_3r(img, t, outsize=outsize, oob=oob),
                             [img, theta])
        elif img.ndim == 2:
            raise ValueError('2d support in transform? [HW]?')


def transform_4r(img, theta, outsize=None, oob=None):
    """

    :param img: [BHWC]
    :param theta: [2,3]
    :param outsize: [H',W'] or [H,W] if none
    :param oob: out of bound value [C,] or [1]
    :return: [BH'W'C]
    """
    assert img.ndim == 4

    if outsize is None:
        # todo improve later
        if None in img.dims[1:3]:
            outsize = tf.shape(img)[1:3]
        else:
            outsize = img.dims[1:3]

    h, w = outsize[0], outsize[1]

    B, H, W, C = img.shapes

    # height, width normalization to (-1, 1)
    # cx = tf.linspace(-1., 1., w)
    # cy = tf.linspace(-1., 1., h)
    cx = tf.linspace(-0.5, 0.5, w)
    cy = tf.linspace(-0.5, 0.5, h)

    xt, yt = tf.meshgrid(cx, cy)

    # target coord
    xyt = tf.stack([xt.flat(), yt.flat(), tf.ones((w * h,))])

    # matching source coord [x; y] [2, pixels]
    xys = theta.dot(xyt)
    # xs, ys = xys.split()  # split along 0 axis

    # reshape to [2, H', W']
    # xys = xys.reshape((2, h, w))

    return sampling_xy(img, xys, outsize, oob=oob)


def transform_3r(img, theta, outsize=None, oob=None):
    """

    :param img: [HWC]
    :param theta: [2,3]
    :param outsize: [H',W'] or [H,W] if none
    :param oob: out of bound value [C,] or [1]
    :return: [H'W'C]
    """
    assert img.ndim == 3

    if outsize is None:
        outsize = img.shapes[:2]  # HWC
        # # todo improve later
        # if None in img.dims[0:2]:
        #     outsize = tf.shape(img)[0:2]
        # else:
        #     outsize = img.dims[0:2]

    h, w = outsize[0], outsize[1]

    H, W, C = img.shapes

    # # height, width normalization to (-1, 1)
    # cx = tf.linspace(-1., 1., w)
    # cy = tf.linspace(-1., 1., h)

    # height, width normalization to (-.5, .5)
    cx = tf.linspace(-0.5, 0.5, w)
    cy = tf.linspace(-0.5, 0.5, h)
    xt, yt = tf.meshgrid(cx, cy)

    # target coord
    xyt = tf.stack([xt.flat(), yt.flat(), tf.ones((w * h,))])

    # matching source coord [x; y] [2, pixels]
    xys = theta.dot(xyt)
    # xs, ys = xys.split()  # split along 0 axis

    # reshape to [2, H', W']
    # xys = xys.reshape((2, h, w))

    return sampling_xy_3r(img, xys, outsize, oob=oob)

# endregion



# region sampling_xy


def sampling_xy(img, xys, outsize=None, oob=None):
    """
    differentiable image sampling (with interpolation)
    :param img: source image [BHWC]
    :param xys: source coord [2, H'*W'] if outsize given
    :param outsize: [H',W'] or None, xys must has rank3
    :return: [B,H',W',C]
    """
    assert img.ndim == 4

    oobv = oob
    if oobv is None:
        oobv = 0.
        # oobv = [0., 0., 0.]
        # oobv = tf.zeros(shape=(img.dims[-1]), dtype=tf.float32)
    oobv = tf.convert_to_tensor(oobv)

    if outsize is None:
        outsize = tf.shape(xys)[1:]
        xys = xys.flat2d()

    B, H, W, C = img.shapes
    WH = tf.stack([W, H]).to_float().reshape((2, 1))

    # XYf = (xys + 1.) * WH * 0.5  # scale to HW coord ( + 1 for start from 0)
    XYf = (xys + 0.5) * WH   # scale to HW coord ( + 1 for start from 0)
    XYS = tf.ceil(XYf)  # left top weight

    # prepare weights
    w00 = XYS - XYf  # [2, p]
    w11 = 1. - w00  # [2, p]

    # get near 4 pixels per pixel
    XYS = XYS.to_int32()  # [2, p]  # todo check xy order
    XYs = XYS - 1
    Xs = tf.stack([XYs[0], XYS[0]])
    Ys = tf.stack([XYs[1], XYS[1]])

    # get mask of outof bound
    # leave option for filling value
    Xi = Xs.clip_by_value(0, W - 1)
    Yi = Ys.clip_by_value(0, H - 1)

    inb = tf.logical_and(Xi.equal(Xs), Yi.equal(Ys))  # [2, p]
    inb = tf.reduce_any(inb, axis=0, keepdims=True)   # all oob? [1, p]-
    inb = inb.expand_dims(2).to_float()

    # get 4 pixels  [B, p, C]
    p00 = getpixel(img, tf.stack([Yi[0], Xi[0]]).T)
    p01 = getpixel(img, tf.stack([Yi[0], Xi[1]]).T)
    p10 = getpixel(img, tf.stack([Yi[1], Xi[0]]).T)
    p11 = getpixel(img, tf.stack([Yi[1], Xi[1]]).T)

    # stacked nearest : [B, 4, p, C]
    near4 = tf.stack([p00, p01, p10, p11], axis=1)

    # XYw : 4 near point weights [4, pixel]
    w4 = tf.stack([w00[1] * w00[0],  # left top
                   w00[1] * w11[0],  # right top
                   w11[1] * w00[0],  # left bottom
                   w11[1] * w11[0]])  # right bottom
    # weighted sum of 4 nearest pixels broadcasting
    w4 = w4.reshape((1, 4, -1, 1))

    interpolated = tf.sum(w4 * near4.to_float(), axis=1)  # [B, p, C]

    # assign oob value
    # fill oob by broadcasting
    oobv = oobv.reshape((1, 1, -1))
    interpolated = interpolated * inb + oobv * (1. - inb)

    output = interpolated.reshape((B, outsize[0], outsize[1], C))
    # reshape [B, p, C] => [B, H', W', C]

    return output


def sampling_xy_3r(img, xys, outsize=None, oob=None):
    """
    differentiable image sampling (with interpolation)
    :param img: source image [HWC]
    :param xys: source coord [2, H'*W'] if outsize given
    :param outsize: [H',W'] or None, xys must has rank3
    :return: [B,H',W',C]
    """
    assert img.ndim == 3

    oobv = oob
    if oobv is None:
        # oobv = tf.zeros(shape=(img.dims[-1]), dtype=tf.float32)  # [0., 0., 0.]
        oobv = 0.
        # oobv = [0., 0., 0.]
    oobv = tf.convert_to_tensor(oobv)

    if outsize is None:
        outsize = tf.shape(xys)[1:]
        xys = xys.flat2d()

    H, W, C = img.shapes
    WH = tf.stack([W, H]).to_float().reshape((2, 1))

    # XYf = (xys + 1.) * WH * 0.5  # scale to HW coord ( + 1 for start from 0)
    XYf = (xys + 0.5) * WH  # * 0.5  # scale to HW coord ( + 1 for start from 0)
    XYS = tf.ceil(XYf)  # left top weight

    # prepare weights
    w00 = XYS - XYf  # [2, p]
    w11 = 1. - w00  # [2, p]

    # get near 4 pixels per pixel
    XYS = XYS.to_int32()  # [2, p]  # todo check xy order
    XYs = XYS - 1
    Xs = tf.stack([XYs[0], XYS[0]])
    Ys = tf.stack([XYs[1], XYS[1]])

    # get mask of outof bound
    # leave option for filling value
    Xi = Xs.clip_by_value(0, W - 1)
    Yi = Ys.clip_by_value(0, H - 1)

    inb = tf.logical_and(Xi.equal(Xs), Yi.equal(Ys))  # [2, p]
    inb = tf.reduce_any(inb, axis=0, keepdims=True)   # all oob? [1, p]-
    # inb = inb.expand_dims(2).to_float()  # [1, p]
    inb = inb.reshape((-1, 1)).to_float()  # [p, 1] 1 for channel

    # get 4 pixels  [p, C]
    p00 = getpixel(img, tf.stack([Yi[0], Xi[0]]).T)
    p01 = getpixel(img, tf.stack([Yi[0], Xi[1]]).T)
    p10 = getpixel(img, tf.stack([Yi[1], Xi[0]]).T)
    p11 = getpixel(img, tf.stack([Yi[1], Xi[1]]).T)

    # stacked nearest : [4, p, C]
    near4 = tf.stack([p00, p01, p10, p11], axis=0)

    # XYw : 4 near point weights [4, pixel]
    w4 = tf.stack([w00[1] * w00[0],  # left top
                   w00[1] * w11[0],  # right top
                   w11[1] * w00[0],  # left bottom
                   w11[1] * w11[0]])  # right bottom
    # weighted sum of 4 nearest pixels broadcasting
    w4 = w4.reshape((4, -1, 1))
    # interpolated = tf.sum(w4 * near4.to_float(), axis=1)  # [p, C]
    interpolated = tf.sum(w4 * near4.to_float(), axis=0)  # [p, C]

    # assign oob value
    # fill oob by broadcasting
    oobv = oobv.reshape((1, -1))  # [p, C]
    interpolated = interpolated * inb + oobv * (1. - inb)

    output = interpolated.reshape((outsize[0], outsize[1], C))
    # reshape [p, C] => [H', W', C]

    return output


# endregion


# region getpixel

def _getpixel_3r(img, ind, name=None):
    assert img.ndim == 3
    # index range? check?
    return tf.gather_nd(img, ind, name=name)


def getpixel(img, ind, name=None):
    """
    no outof bound checking
    :param img: [NHWC] or [HWC]
    :param ind: [p, ij(2)] (height index, width index)
    :param name:
    :return: [N, p, C] or [p, C]
    """
    if img.ndim == 3:
        return _getpixel_3r(img, ind, name=name or 'getpixel')
    elif img.ndim == 4:
        return tf.map_fn(lambda x: _getpixel_3r(x, ind),
                         img, name=name or 'getpixel')
    else:
        raise ValueError("need 2d support in getpixel?")

# endregion

# todo add more example

if __name__ == '__main__':
    from skimage.data import coffee
    import matplotlib.pyplot as plt

    img = coffee()
    img = tf.convert_to_tensor(img / 255., dtype=tf.float32)
    img = tf.expand_dims(img, 0)

    theta = tf.convert_to_tensor([[0.5, 0., 0.], [0., 0.5, 0.]])
    t = transform(img, theta)

    plt.subplot(1, 2, 1)
    plt.imshow(img[0].eval())

    plt.subplot(1, 2, 2)
    plt.imshow(t[0].eval())

    plt.show()
    print('done')

