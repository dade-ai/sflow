# -*- coding: utf-8 -*-
import tensorflow as tf
from six import wraps

# region optimizer or callable(lr)


def _deco_functional_optimizer(orgoptim):
    def wrap(fn):

        @wraps(fn)
        def wrapped(lr=None, **kwargs):

            @wraps(orgoptim)
            def _partial(learning_rate):
                return orgoptim(learning_rate, **kwargs)

            if lr is not None:
                return _partial(lr)
            else:
                return _partial

        return wrapped
    return wrap


@_deco_functional_optimizer(tf.train.GradientDescentOptimizer)
def SGD(lr=None, **kwargs):
    pass


@_deco_functional_optimizer(tf.train.AdadeltaOptimizer)
def Adadelta(lr=None, **kwargs):
    pass


@_deco_functional_optimizer(tf.train.AdagradOptimizer)
def Adagrad(lr=None, **kwargs):
    pass


@_deco_functional_optimizer(tf.train.MomentumOptimizer)
def Momentum(lr=None, **kwargs):
    pass


@_deco_functional_optimizer(tf.train.AdamOptimizer)
def Adam(lr=None, **kwargs):
    pass

@_deco_functional_optimizer(tf.train.FtrlOptimizer)
def Ftrl(lr=None, **kwargs):
    pass


@_deco_functional_optimizer(tf.train.RMSPropOptimizer)
def RMSProp(lr=None, **kwargs):
    pass


@_deco_functional_optimizer(tf.train.ProximalAdagradOptimizer)
def ProximalAdagrad(lr=None, **kwargs):
    pass


@_deco_functional_optimizer(tf.train.ProximalGradientDescentOptimizer)
def ProximalGradientDescent(lr=None, **kwargs):
    pass


@_deco_functional_optimizer(tf.train.AdagradDAOptimizer)
def AdagradDA(lr=None, **kwargs):
    pass


from . import ioptimizer
@_deco_functional_optimizer(ioptimizer.MaxPropOptimizer)
def MaxProp(lr=None, **kwargs):
    pass


# endregion

