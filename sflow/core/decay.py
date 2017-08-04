# -*- coding: utf-8 -*-
from __future__ import absolute_import
import tensorflow as tf
from .icore import (get_global_step)


def exponential(decay_steps, decay_rate, staircase=False, lr=None, name=None):

    def exponential_decay(learning_rate, global_step=None):
        global_step = global_step or get_global_step()
        return tf.train.exponential_decay(learning_rate, global_step,
                                          decay_steps, decay_rate,
                                          staircase=staircase, name=name)
    if lr:
        return exponential_decay(lr)
    return exponential_decay


def piecewise(boundaries, lr=None, name=None):

    def piecewise_constant(values, global_step=None):
        global_step = global_step or get_global_step()
        return tf.train.piecewise_constant(global_step, boundaries, values, name=name)

    if lr:
        return piecewise_constant(lr)

    return piecewise_constant


def linear(decay_steps, lr_end=0.0001,
           cycle=False, lr=None, name=None):
    return polynomial(decay_steps, end_learning_rate=lr_end,
                      cycle=cycle, lr=lr, name=name, power=1.0)


def polynomial(decay_steps, end_learning_rate=0.0001, power=1.0,
               cycle=False, lr=None, name=None):

    def polynomial_decay(learning_rate, global_step=None):
        global_step = global_step or get_global_step()
        return tf.train.polynomial_decay(learning_rate, global_step,
                                         decay_steps, end_learning_rate=end_learning_rate,
                                         power=power, cycle=cycle, name=name)
    if lr:
        return polynomial_decay(lr)

    return polynomial_decay


def natural_exp(decay_steps, decay_rate, staircase=False, lr=None, name=None):

    def natural_exp_decay(learning_rate, global_step=None):
        global_step = global_step or get_global_step()
        return tf.train.natural_exp_decay(learning_rate, global_step,
                                          decay_steps, decay_rate, staircase=staircase, name=name)

    if lr:
        return natural_exp_decay(lr)

    return natural_exp_decay


def inverse_time(decay_steps, decay_rate, staircase=False, lr=None, name=None):

    def inverse_time_decay(learning_rate, global_step=None):
        global_step = global_step or get_global_step()
        return tf.train.inverse_time_decay(learning_rate, global_step,
                                           decay_steps, decay_rate, staircase=staircase, name=name)
    if lr:
        return inverse_time_decay(lr)

    return inverse_time_decay

