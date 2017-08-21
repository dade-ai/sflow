# -*- coding: utf-8 -*-
from __future__ import absolute_import

from functools import wraps

import tensorflow as tf
from tensorflow.python.framework import ops

from snipy.basic import (patchmethod, tuple_args, optional_str)
from .isummary import summary_gradient
from .icore import get_global_step
from . import logg as logg

# region optimizer patch methods


@patchmethod(tf.train.Optimizer, name='get_slot_variables')
def _get_slot_variables(opt):
    """
    optimizer variable list (공통)
    :param tf.train.Optimizer opt:
    :return list[tf.Tensor]:
    """

    def _opt_variables(o):
        # noinspection PyProtectedMember
        for d in o._slots.values():
            for v in d.values():
                yield v

    return list(_opt_variables(opt))


@patchmethod(tf.train.Optimizer, name='get_variables')
def _get_variables(opt):
    """
    optimizer variable list (공통)
    :param tf.train.Optimizer opt:
    :return list[tf.Tensor]:
    """
    return _get_slot_variables(opt)


# noinspection PyProtectedMember
@patchmethod(tf.train.AdamOptimizer, name='get_variables')
def _get_adam_variables(opt):
    return opt.get_slot_variables() + list(opt._get_beta_accumulators())


@patchmethod(tf.train.Optimizer, name='init_variables')
def _init_variables(opt):
    # patched function

    vlist = opt.get_variables()  # call patched method
    init = tf.variables_initializer(vlist, name='optimizer_vars_init')
    init.run()


@patchmethod(tf.train.Optimizer, name='train')
def _train(optim, loss, scope=None, global_step=None, **kwargs):
    """
    todo add example
    :param optim:
    :return:
    """
    # make global_step as mandatory
    global_step = global_step or get_global_step()
    return _minimize(optim, loss, scope=scope, global_step=global_step, **kwargs)


@patchmethod(tf.train.Optimizer)
def _get_learning_rate_tensor(optim):
    try:
        return optim._learning_rate_tensor
    except AttributeError:
        return optim._lr_t
    except:
        raise AttributeError('what optimizer? learning rate tensor?')


@patchmethod(tf.train.Optimizer, name='minimize')
def _minimize(optim, loss, global_step=None, var_list=None,
              gate_gradients=tf.train.Optimizer.GATE_OP, aggregation_method=None,
              colocate_gradients_with_ops=False, name=None,
              grad_loss=None, scope=None, summary=False):
    """
    optimizer.minimize wrapper
    :param loss: Tensor
    :param optim: Optimizer
    :param scope: str, variables name scope strings
    :param kwargs: optimizer args
    :return:
    """
    # todo : add output metric information.?

    # get variable list
    if scope:
        assert not var_list  # mutually exclusive parameter
        if not isinstance(scope, (list, tuple)):
            scope = [scope]
        var_list = []
        for s in scope:
            var_list.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=s))

    grads_and_vars = optim.compute_gradients(loss, var_list=var_list,
                                             gate_gradients=gate_gradients,
                                             aggregation_method=aggregation_method,
                                             colocate_gradients_with_ops=colocate_gradients_with_ops,
                                             grad_loss=grad_loss)

    # from Optimizer.minimize
    vars_with_grad = [v for g, v in grads_and_vars if g is not None]
    if not vars_with_grad:
        raise ValueError(
            "No gradients provided for any variable, check your graph for ops"
            " that do not support gradients, between variables %s and loss %s." %
            ([str(v) for _, v in grads_and_vars], loss))

    # add summary
    if summary is True:
        for g, v in grads_and_vars:
            summary_gradient(v, g)

    # todo@dade : check this (bn)
    # scope filtering

    updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(updates):
        update_op = optim.apply_gradients(grads_and_vars, global_step=global_step, name=name)

    # updates = []
    # if scope:
    #     for s in scope:
    #         updates.extend(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=s))
    # else:
    #     updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #
    # if updates:
    #     with tf.control_dependencies(updates):
    #         update_op = optim.apply_gradients(grads_and_vars, global_step=global_step, name=name)
    # else:
    #     update_op = optim.apply_gradients(grads_and_vars, global_step=global_step, name=name)

    # update_op = optim.apply_gradients(grads_and_vars, global_step=global_step, name=name)

    # add learning_rate scalar history with proper naming
    lr = _get_learning_rate_tensor(optim)
    if scope is None:
        lossname = loss.name
    elif not isinstance(scope, (list, tuple)):
        lossname = scope
    else:
        lossname = '_'.join(scope)
    tf.summary.scalar('learning_rate/' + lossname, lr)

    # check slot variables
    # tf.add_to_collection('OPTIMIZER_VAR', optim.get_variables())

    return update_op

# endregion


# region optim util
def minimize(loss, lr=0.001, optim=None, decay=None, scope=None):
    """
    example::

        trainop = tf.minimize(loss, lr=0.001,
                              decay=tf.decay.exponential(100, 0.99),
                              optim=tf.optim.Adam(**option_without_lr))

    :param loss:
    :param scope:
    :param lr:
    :param optim:
    :param decay:
    :return:
    """
    # todo : add some example use cases
    # if scope is None:
    #     pass
    from .icore import get_global_step
    from .optim import Adam
    optim = optim or Adam()

    if isinstance(optim, tf.train.Optimizer):
        if lr != 0.001:
            raise ValueError('optimizer already has a learning rate, invalid argument lr:{0}'.format(lr))
        if decay:
            raise ValueError('optimizer already has a learning rate, invalid argument decay:{0}'.format(str(decay)))
    else:
        # optim is a callable with optim(lr, [global_step])
        if decay:
            lr = decay(lr, global_step=get_global_step())
        optim = optim(lr)

    # noinspection PyArgumentList
    trainop = optim.minimize(loss, scope=scope)

    return trainop


@tuple_args
def trains_increase_step(trainops=None, global_step=None, name=None):
    """
    example::

        blabla

    return operation of training then increase 1 to global_step
    :param trainops:
    :param global_step:
    :param name:
    :return:
    """
    from tensorflow.python.framework.ops import colocate_with
    from .icore import get_global_step
    trainoplist = tf.get_collection_ref(tf.GraphKeys.TRAIN_OP)

    if trainops is None or len(trainops) == 0:
        trainops = trainoplist

    if trainops is None or len(trainops) == 0:
        raise ValueError('no trainops')

    global_step = global_step or get_global_step()

    with tf.control_dependencies(trainops):
        with colocate_with(global_step):
            apply_updates = tf.assign_add(global_step, 1, name=name).op

    if apply_updates not in trainoplist:
        trainoplist.append(apply_updates)

    return apply_updates


def increase_step(global_step=None, name=None):
    """
    example::

        blabla

    return operation of training then increase 1 to global_step
    :param global_step:
    :param name:
    :return:
    """
    global_step = global_step or get_global_step()
    apply_updates = tf.assign_add(global_step, 1, name=name).op
    return apply_updates

# endregion

# region optimizers


# noinspection PyAbstractClass
class MaxPropOptimizer(tf.train.Optimizer):
    r"""Optimizer that implements the MaxProp algorithm by buriburisuri@gmail.com.
    """
    def __init__(self, learning_rate=0.001, beta2=0.999, use_locking=False, name="MaxProp"):
        super(MaxPropOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta2 = beta2

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta2_t = None

    def _prepare(self):
        self._lr_t = tf.convert_to_tensor(self._lr, name="learning_rate")
        self._beta2_t = tf.convert_to_tensor(self._beta2, name="beta2")

    def _create_slots(self, var_list):
        # Create slots for the second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)

    def _apply_dense(self, grad, var):
        lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
        beta2_t = tf.cast(self._beta2_t, var.dtype.base_dtype)
        if var.dtype.base_dtype == tf.float16:
            eps = 1e-7  # Can't use 1e-8 due to underflow -- not sure if it makes a big difference.
        else:
            eps = 1e-8

        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta2_t * m + eps, tf.abs(grad)))
        g_t = grad / m_t

        var_update = tf.assign_sub(var, lr_t * g_t)
        return tf.group(*[var_update, m_t])

    def _apply_sparse(self, grad, var):
        return self._apply_dense(grad, var)


GradientDescentOptimizer = tf.train.GradientDescentOptimizer
AdadeltaOptimizer = tf.train.AdadeltaOptimizer
AdagradOptimizer = tf.train.AdagradOptimizer
MomentumOptimizer = tf.train.MomentumOptimizer
AdamOptimizer = tf.train.AdamOptimizer
FtrlOptimizer = tf.train.FtrlOptimizer
RMSPropOptimizer = tf.train.RMSPropOptimizer
ProximalAdagradOptimizer = tf.train.ProximalAdagradOptimizer
ProximalGradientDescentOptimizer = tf.train.ProximalGradientDescentOptimizer
AdagradDAOptimizer = tf.train.AdagradDAOptimizer


# endregion

# region gradient

@optional_str
def register_gradient(name=None):

    def _decorate(fun):

        @wraps(fun)
        def _grad_wrap(op, grad):
            return [fun(grad)]

        fname = name or fun.__name__
        if fname in ops._gradient_registry._registry:
            logg.warn('Already registered gradfun: [{}]'.format(fname))
        else:
            ops._gradient_registry.register(_grad_wrap, fname)
        return fun

    return _decorate


def grad_apply(gradfun, x, name=None, gname=None):
    """
    :param gradfun: new_grad = gradfun(grad)
    :param x: tensor or tensorlist
    :param name:
    :param gname:
    :return: x
    """
    grad_name = gname or gradfun.__name__

    # @ops.RegisterGradient(grad_name)
    # def _grad_apply(grad):
    #     return [gradfun(grad)]

    @register_gradient(grad_name)
    @wraps(gradfun)
    def _gradfun(g):
        return tf.identity(gradfun(g), name=gname)

    g = tf.get_default_graph()
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(x, name=name)

    return y


def gradflip(x):
    """
    gradient flip
    example
        In [17]: a = tf.ones((3,2))
        In [18]: b = tf.square(a)
        In [19]: g1 = tf.gradients(b, a)
        In [20]: g1[0].eval()
        Out[20]:
        array([[ 2.,  2.],
               [ 2.,  2.],
               [ 2.,  2.]], dtype=float32)
        In [21]: flip = tf.gradflip(b)
        In [22]: gflip = tf.gradients(flip, a)
        In [23]: gflip[0].eval()
        Out[23]:
        array([[-2., -2.],
               [-2., -2.],
               [-2., -2.]], dtype=float32)

    :param x:
    :return:
    """
    @wraps(gradflip)
    def _gradflip(g):
        return -g

    return grad_apply(_gradflip, x)


def gradscale(x, scale):
    @wraps(gradscale)
    def _gradscale(g):
        return g * scale
    return grad_apply(_gradscale, x)

# endregion

# region manipulate weights

def clip_variables(min, max, trainop=None, scope=None):
    """
    clip variables
    :param min:
    :param max:
    :param trainop:
    :param scope:
    :return:
    """
    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    assert len(vars)

    trainops = trainop if isinstance(trainop, (tuple, list)) or trainop is None else [trainop]

    with tf.control_dependencies(trainops):
        varclip = [tf.assign(v, tf.clip_by_value(v, min, max)).op for v in vars]
        clipop = tf.group(*varclip)

    # add train op if not in
    trainoplist = tf.get_collection_ref(tf.GraphKeys.TRAIN_OP)
    if clipop not in trainoplist:
        trainoplist.append(clipop)

    return clipop


# endregion

