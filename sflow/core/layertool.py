# -*- coding: utf-8 -*-
from __future__ import absolute_import

import tensorflow as tf
from six import wraps

from .defaults import (call_with_default_context)
from snipy.basic import (patch)


def layer(fn):

    @wraps(fn)
    def wrapped(*args, **kwargs):
        n = kwargs.pop('name', None)
        sc = kwargs.pop('scope', None) or fn.__name__

        with tf.variable_scope(n, sc):
            return call_with_default_context(fn, *args, **kwargs)

    patch.method([tf.Tensor, tf.Variable], wrapped)

    return wrapped

