# -*- coding: utf-8 -*-
from __future__ import absolute_import
import tensorflow as tf


# region summary function


def summary_losses(tensors, names=None):
    if names is None:
        names = [None] * len(tensors)
    for t, n in zip(tensors, names):
        summary_loss(t, name=n)
    return tensors


def summary_loss(tensor, name=None):

    name = 'loss/' + (name or tensor.name.replace(':', '_'))
    with tf.name_scope(name) as s:
        tf.summary.scalar(s, tensor)

    return tensor


def summary_learning_rate(tensor, name=None):
    # defaults
    name = 'learning_rate/' + (name or tensor.name)
    tf.summary.scalar(name, tensor)

    return tensor


def summary_activations(tensors, names=None):
    if names is None:
        names = [None] * len(tensors)
    for t, n in zip(tensors, names):
        summary_activation(t, name=n)
    return tensors


def summary_activation(tensor, name=None):
    # defaults
    name = 'activation/' + (name or tensor.name)
    # summary statistics
    # tf.summary.scalar(name + '/norm', tf.global_norm([tensor]))
    tf.summary.scalar(name + '/ratio', tensor.greater(0.).to_float().mean())
    tf.summary.scalar(name + '/max', tensor.max())
    tf.summary.scalar(name + '/min', tensor.min())
    tf.summary.histogram(name, tensor)

    return tensor


def summary_range(tensor, name=None):
    # defaults
    name = 'activation/' + (name or tensor.name)
    tf.summary.scalar(name + '/max', tensor.max())
    tf.summary.scalar(name + '/min', tensor.min())

    return tensor


def summary_gradient(tensor, gradient, name=None):

    if gradient is None:
        return
    name = 'gradient/' + (name or tensor.name)
    # summary statistics
    tf.summary.scalar(name + '/norm', tf.global_norm([gradient]))
    tf.summary.histogram(tensor.name, gradient)

    return tensor


def summary_grad(ys, tensor, name=None):
    gradient = tf.gradients(ys, tensor)

    name = 'gradient/' + (name or tensor.name)
    tf.summary.histogram(name, gradient)
    return gradient


def summary_param(tensor, name=None):
    name = 'param/' + (name or tensor.name)
    # summary statistics
    tf.summary.scalar(name + '/norm', tf.global_norm([tensor]))
    tf.summary.histogram(tensor.name, tensor)

    return tensor


def summary_image(tensor, name=None, sizedown=None, max=3):
    org = tf.identity(tensor)
    name = name or tensor.name
    if sizedown is not None:
        tensor = tensor.sizedown(sizedown)
    tf.summary.image(name, tensor, max_outputs=max)

    return org


def summary_audio(tensor, sample_rate=16000, name=None):
    name = name or tensor.name
    tf.summary.audio(name, tensor, sample_rate)

    return tensor

