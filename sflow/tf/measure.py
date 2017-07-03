# -*- coding: utf-8 -*-
import sflow.core as tf

# region metric


@tf.op_scope
def total_variance_iso(t, name=None):
    """
    https://en.wikipedia.org/wiki/Total_variation_denoising
    mean total variation
    :param t: [N,h,w,c]
    :return: mean not sum
    """
    assert t.ndim == 4
    dx = t[:, :-1, :-1, :] - t[:, 1:, :-1, :]
    dy = t[:, :-1, :-1, :] - t[:, -1:, 1:, :]

    return tf.sqrt(dx.square() + dy.square(), name=name).mean()

