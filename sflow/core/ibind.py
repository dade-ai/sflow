# -*- coding: utf-8 -*-
from __future__ import absolute_import

import tensorflow as tf
from contextlib import contextmanager
from .logg import logg
import snipy.decotool as decotool

__all__ = ['bind', 'bind_scope', '_']


@contextmanager
def bind_scope(**funmap):
    """
    example::

        acts2 = []
        with tf.bind_scope(keep2=tf.bind(tf.keep, keepto=acts2)):
                h = x
                h = h.conv(64).relu().keep()
                h = h.conv(64).relu().keep2()

    :param funmap:
    :return:
    """

    # remember original
    saved = dict()
    for k, v in funmap.items():
        old = getattr(tf.Tensor, k, None)
        if old is not None:
            logg.warn('function[{}] already exist in Tensor'.format(k))
            saved[k] = old

    # patch method with new
    for k, v in funmap.items():
        setattr(tf.Tensor, k, v)
    yield
    # remove and restore
    for k, v in funmap.items():
        delattr(tf.Tensor, k)
    for k, v in saved.items():
        setattr(tf.Tensor, k, v)


def bind(fun, *argsfrom2, **kwbind):
    """
    bind function with first argument as a tensor
    :return: callable bound
    """
    import snipy.decotool as decotool
    _ = decotool.bind.placeholder
    argsbind = [_] + list(argsfrom2)
    return decotool.bind(fun, *argsbind, **kwbind)

_ = bind._ = bind.placeholder = decotool.bind.placeholder

