# -*- coding: utf-8 -*-

import tensorflow as tf

from snipy.basic import (optional_str, wraps)
# from .common23 import *


@optional_str
def op_scope(name=None):
    """
    decorator version of tf.name_scope
    example::

        @op_scope('MyOp')
        def my_op(a, b, c, name=None):
          a = tf.convert_to_tensor(a, name="a")
          b = tf.convert_to_tensor(b, name="b")
          c = tf.convert_to_tensor(c, name="c")
          return foo_op(..., name=scope)
        =>
        def my_op(a, b, c, name=None):
          with tf.name_scope(name, "MyOp", [a, b, c]) as scope:
            a = tf.convert_to_tensor(a, name="a")
            b = tf.convert_to_tensor(b, name="b")
            c = tf.convert_to_tensor(c, name="c")
            # Define some computation that uses `a`, `b`, and `c`.
            return foo_op(..., name=scope)

    :param name: str, default name in tf.name_scope if None, use function name
    :return: wrapped function
    """

    def wrap(fun):
        dname = name or fun.__name__

        @wraps(fun)
        def wrapped(*args, **kwargs):
            n = kwargs.pop('name', None)
            with tf.name_scope(n, default_name=dname, values=args) as sc:
                res = fun(*args, **kwargs)

                if isinstance(res, (tuple, list)):
                    return tuple(tf.identity(t, name='{}_{}'.format(sc, i))
                                 for i, t in enumerate(res))
                else:
                    return tf.identity(res, name=sc)

        return wrapped

    return wrap


@optional_str
def scope(name=None):
    """
    decorator version of tf.variable_scope
    example::

        @scope('MyOp')
        def my_op(a, b, c, name=None):
          a = tf.convert_to_tensor(a, name="a")
          b = tf.convert_to_tensor(b, name="b")
          c = tf.convert_to_tensor(c, name="c")
          return foo_op(..., name=scope)
        =>
        def my_op(a, b, c, name=None):
          with tf.variable_scope(name or 'my_op') as vscope:
              a = tf.convert_to_tensor(a, name="a")
              b = tf.convert_to_tensor(b, name="b")
              c = tf.convert_to_tensor(c, name="c")
              # Define some computation that uses `a`, `b`, and `c`.
              return foo_op(..., name=scope)

    :param name: str, default name in tf.name_scope if None, use function name
    :return: wrapped function
    """

    def wrap(fun):
        return _Scoped(fun, name or fun.__name__)

    return wrap


class _Scoped(object):

    def __init__(self, fun, name):
        self.name = name
        self.fun = fun

    def __call__(self, *args, **kwargs):
        n = kwargs.pop('name', None)
        with tf.variable_scope(n, self.name, args) as vscope:
            return self.fun(*args, **kwargs)


@optional_str
def reuse_scope(name=None):
    """
    decorator version of tf.name_scope with tf.make_template
    for sharing variables, cf. Scope.to_template
    example::

        @reuse_scope('MyOp')
        def my_op(a, b, c, name=None):
          a = tf.convert_to_tensor(a, name="a")
          b = tf.convert_to_tensor(b, name="b")
          c = tf.convert_to_tensor(c, name="c")
          return foo_op(..., name=scope)
        =>
        def my_op(a, b, c, name=None):
            with tf.name_scope(name, 'my_op', [a, b, c]) as scope:
              a = tf.convert_to_tensor(a, name="a")
              b = tf.convert_to_tensor(b, name="b")
              c = tf.convert_to_tensor(c, name="c")
              # Define some computation that uses `a`, `b`, and `c`.
              return foo_op(..., name=scope)
        return tf.make_template(name or 'my_op', my_op)

    :param name: str, default name in tf.name_scope if None, use function name
    :return: wrapped function
    """

    def wrap(fun):
        n = name or fun.__name__
        return tf.make_template(n, fun)

    return wrap


def reusable(fun, name=None):
    """
    function version for reuse_scope
    see also tf.make_template
    :param fun:
    :param name:
    :return:
    """
    # from tensorflow.python.ops.template import Template

    if isinstance(fun, _Scoped):
        return tf.make_template('', fun)
    else:
        return tf.make_template(name or fun.__name__, fun)


# endregion



