# -*- coding: utf-8 -*-
import tensorflow as tf


def random_choice(shape, values=None, p=0.5):
    mask = tf.random_uniform(shape, dtype=tf.float32) >= p
    if values is None:
        return mask
    else:
        out = tf.where(mask, tf.constant(values[0], shape=shape), tf.constant(values[1], shape=shape))
    return out


