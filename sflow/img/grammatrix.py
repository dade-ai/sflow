# -*- coding: utf-8 -*-
import sflow.core as tf
import numpy as np


def grammatrix(t):
    """
    grammatrix
    :param t: [batch, h, w, channel]
    :return: [batch, channel, channel]
    """
    # for all batch
    #   res[batch] = [channel, hw] dot [hw, channel]

    # assert t.ndim == 4

    # reshape first
    dim = t.dims
    r = t.reshape((dim[0], dim[1] * dim[2], dim[3]))
    r_t = r.transpose(0, 2, 1)
    # batch_matmul (batch dot)

    return r_t.dot(r)
