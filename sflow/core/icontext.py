# -*- coding: utf-8 -*-
import tensorflow as tf

from snipy.basic import (optional_str, wraps)
# from .common23 import *
from .logg import logg


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
        self.__name__ = name
        self.fun = fun

    def __call__(self, *args, **kwargs):
        n = kwargs.pop('name', None)
        with tf.variable_scope(n, self.name, args) as vscope:
            return self.fun(*args, **kwargs)

# tf.make_template()
# from tensorflow.python.framework.ops.template import Template
from tensorflow.python.ops.template import Template


class TemplateEx(Template):

    def __init__(self, name, func, **kwargs):
        self.__name__ = func.__name__
        super(TemplateEx, self).__init__(name, func, **kwargs)

    def __call__(self, *args, **kwargs):
        reuse = kwargs.pop('reuse', True)
        if reuse is not True:
            return self._func(*args, **kwargs)
        else:
            if self._variables_created:
                logg.info('reusing template: {}'.format(self.__name__))
        return super(TemplateEx, self).__call__(*args, **kwargs)

# #keep original function
# _make_template = tf.make_template


def make_template(func, name=None, create_scope_now_=False, unique_name_=None,
                  custom_getter_=None, **kwargs):
    import functools
    # modified from tensorflow source
    name = name if name is not None else func.__name__

    if kwargs:
        func = functools.partial(func, **kwargs)
    return TemplateEx(name, func, create_scope_now=create_scope_now_,
                      unique_name=unique_name_, custom_getter=custom_getter_)


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
        return make_template(fun, name)

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
        return make_template(fun, '')
    else:
        return make_template(fun, name)


# endregion



