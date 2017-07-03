# -*- coding: utf-8 -*-
import sflow.core as tf
# import numpy as np


def load_lua(fpath, **kwargs):
    """
    Loads the given t7 file using default settings; kwargs are forwarded
    to `T7Reader`.
    :param fpath:
    :param kwargs:
    :return:
    """
    from torch.utils.serialization import load_lua
    return load_lua(fpath, **kwargs)


def load_lua_params(fpath, **kwargs):
    from snipy.concurrent import threaded

    th = threaded(load_lua_params_impl, fpath, **kwargs)
    return th.result


def load_lua_params_impl(fpath, **kwargs):
    """
    load torch7 model, collect weight and bias
    :param fpath:
    :return: weights, infos
    :rtype list(numpy), list(str)
    """
    model = load_lua(fpath, **kwargs)

    params = []
    infos = []
    for m in model.modules:
        # skip first image mean extraction part
        desc = repr_torch(m)
        try:
            w = m.weight
        except AttributeError:
            tf.logg.warn('skip layer [{}]'.format(desc))
            continue

        w, desc = get_weight(m)

        slog = 'layer [{}].w[{}]'.format(desc, w.shape)

        params.append(w)
        infos.append(desc + '.w')

        try:
            b = m.bias
        except AttributeError:
            tf.logg.info('{}'.format(slog))
            continue

        b = b.numpy().astype('float32')
        tf.logg.info('{}.b[{}]'.format(slog, b.shape))
        params.append(b)
        infos.append(desc + '.b')

    return params, infos


def get_weight(m):
    """
    :param m: TorchObject
    :return: weight reshaped
    """
    w = m.weight
    w = w.numpy().astype('float32')
    tname = repr_torch(m)

    if w.ndim == 4:
        # conv?
        w = w.transpose((2, 3, 1, 0))
    elif 'SpatialConvolutionMM' in tname:
        # weight : (outputplane, ???) => (h,w,inplane, outplane)
        # see: line 20 https://github.com/torch/nn/blob/master/SpatialConvolutionMM.lua
        # self.weight = torch.Tensor(nOutputPlane, nInputPlane*kH*kW)
        # self.bias = torch.Tensor(nOutputPlane)
        w = w.reshape((m.nOutputPlane, m.nInputPlane, m.kH, m.kW))
        w = w.transpose((2, 3, 1, 0))
        tname = '{}.pad[{}]'.format(tname, (m.padH, m.padW))
    elif 'SpatialConvolution' in tname and w.ndim == 4:
        # conv
        w = w.transpose((2, 3, 1, 0))
    elif 'Linear' in tname:
        w = w.transpose((1, 0))

    # else:
    #     tf.logg.warn('no conv [{}].w[{}]'.format(tname, w.shape))

    return w, tname


def repr_torch(m):
    try:
        tname = m.torch_typename()
    except AttributeError:
        return m.__repr__()
    return tname

