# -*- coding: utf-8 -*-
import sflow.core as tf
import snipy.basic as basic

# for alpha blending by tensorflow
# combine RGBA, RGBA of (0., 1.)


@basic.tuple_args
def alpha_composite(images, order=None, **whileopt):
    """
    example
    alpha_composite([fg1, ... bg]) shape [B,H,W,C], C==4
    alpha_composite(fg1, ... bg) == alpha_composite([fg1, ... bg])
    alpha_composite(batch_layered_images)  # shape [B, Layer, H, W, C], C= 4

    :param images: order from fg to bg [bhwc] assert c == 4 (RGBA)
    :param order: None for guess, 'BL' or 'LB', meaning of 2 leading axis. check this value and decide order if arg is a tensor.
    :return: composed images
    """
    # https://en.wikipedia.org/wiki/Alpha_compositing
    # alpha_out = alpha_fg + alpha_bg * ( 1 - alpha_fg)
    # if alpha_out:
    #   rgb_out = (rgb_fg * alpha_fg + rgb_bg * alpha_bg * ( 1 - alpha_fg)) / alpha_out
    # else:
    #   rgb_out = 0
    if len(images) == 1:
        # alpha_composite for layered tensor
        # images : shape [ Layer, B, H, W, C]
        images = images[0]
        # assert tensor..
        if images.ndim == 4 and order is None:
            # [LHWC]  Layer, H, W, C
            pass
        elif images.ndim == 5 and order == 'BL':
            images = images.transpose([1, 0, 2, 3, 4])
        else:
            raise ValueError('check image tensor and order arguments')

    # stack all images
    # assert all rgba
    # [Layer, B, H, W, C]
    else:
        images = tf.stack(images)

    def step(bg, fg):
        alpha_fg = fg[..., -1:]
        alpha_bg = bg[..., -1:]
        rgb_fg = fg[..., :-1]
        rgb_bg = bg[..., :-1]
        alpha = alpha_fg + alpha_bg * (1. - alpha_fg)
        visible = tf.not_equal(alpha, 0.)

        # nan check
        alphadiv = tf.select(visible, alpha, 1.)

        rgb = tf.select(visible, (rgb_fg * alpha_fg + rgb_bg * alpha_bg * (1. - alpha_fg))/alphadiv, 0.)
        out = tf.concat(-1, [rgb, alpha])
        return out

    composed = tf.foldleft(step, images, **whileopt)
    return composed


@basic.tuple_args
def composite(images, order=None, **whileopt):
    """
    example
    composite([fg1, ... bg]) shape [B,H,W,C], C==3
    composite(fg1, ... bg) == alpha_composite([fg1, ... bg])
    composite(batch_layered_images)  # shape [B, Layer, H, W, C], C= 3

    :param images: order from fg to bg [bhwc] assert c == 4 (RGBA)
    :param order: None for guess, 'BL' or 'LB', meaning of 2 leading axis. check this value and decide order if arg is a tensor.
    :return: composed images
    """
    # https://en.wikipedia.org/wiki/Alpha_compositing
    # alpha_out = alpha_fg + alpha_bg * ( 1 - alpha_fg)
    # if alpha_out:
    #   rgb_out = (rgb_fg * alpha_fg + rgb_bg * alpha_bg * ( 1 - alpha_fg)) / alpha_out
    # else:
    #   rgb_out = 0
    if len(images) == 1:
        # alpha_composite for layered tensor
        # images : shape [ Layer, B, H, W, C]
        images = images[0]
        # assert tensor..
        if images.ndim == 4 and order is None:
            # [LHWC]  Layer, H, W, C
            pass
        elif images.ndim == 5 and order == 'BL':
            images = images.transpose([1, 0, 2, 3, 4])
        else:
            raise ValueError('check image tensor and order arguments')

    # stack all images
    # assert all rgba
    # [Layer, B, H, W, C]
    else:
        images = tf.stack(images)

    def step(bg, fg):
        transparent = tf.equal(fg, 0.).all(axis=-1, keepdims=True)
        res = tf.select(transparent, bg, fg)
        return res

    composed = tf.foldleft(step, images, **whileopt)
    return composed

