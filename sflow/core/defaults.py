# -*- coding: utf-8 -*-
from six import wraps
from collections import (MutableMapping, defaultdict, Mapping)

from tensorflow.python.framework.ops import _DefaultStack
# import tensorflow as tf


class Dic(MutableMapping):

    def __init__(self, *args, **kwargs):
        self.__dict__.update(*args, **kwargs)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __delitem__(self, key):
        del self.__dict__[key]

    def __getattr__(self, key):
        return None

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__dict__.__repr__()

    def __add__(self, other):
        res = Dic(self.__dict__)
        for k, v in other.items():
            if k not in res.__dict__ or res.__dict__[k] is None:
                res.__dict__[k] = v
        return res

    def __mul__(self, other):
        res = Dic(self.__dict__)
        for k, v in other.items():
            res.__dict__[k] = v
        return res

    def as_default(self):
        """ Returns a context manager that makes this `opts` as default.
        if already context merge defaults
        """
        opt = get_default_opts()
        if opt is None:
            return _default_opts_stack.get_controller(self)
        else:
            merge = Dic(**opt)
            merge.update(self)
            return _default_opts_stack.get_controller(merge)

    def to_default(self, fn):
        """ default arg binding to specific function """
        opt = get_default_opts(fn)
        if opt is None:
            return _default_opts_stack_map[fn].get_controller(self)
        else:
            merge = Dic(**opt)
            merge.update(self)
            return _default_opts_stack_map[fn].get_controller(merge)

    def eval(self, sess=None, feed_dict=None, options=None, run_metadata=None, **kwargs):
        # call if all tensors
        from .icore import default_session
        sess = sess or default_session()
        return sess.run(self, feed_dict=feed_dict, options=options, run_metadata=run_metadata, **kwargs)

    @classmethod
    def dict_to_dic(cls, data):
        """
        convert recursively
        :param data:
        :return:
        """
        if isinstance(data, (tuple, list)):
            t = type(data)
            return t(cls.dict_to_dic(d) for d in data)
        elif isinstance(data, Mapping):
            return Dic((k, cls.dict_to_dic(v)) for k, v in data.items())
        else:
            return data

    @classmethod
    def dic_to_dict(cls, data):
        """
        convert recursively
        :param data:
        :return:
        """
        if isinstance(data, (tuple, list)):
            t = type(data)
            return t(cls.dic_to_dict(d) for d in data)
        elif isinstance(data, Mapping):
            return dict((k, cls.dic_to_dict(v)) for k, v in data.items())
        else:
            return data


# aliasing
dic = Dic


class _DefaultOptsStack(_DefaultStack):
    """A thread-local stack of objects for providing an implicit default graph."""
    def __init__(self):
        super(_DefaultOptsStack, self).__init__()
        self._global_default_opts = None

    def get_default(self):
        """Override that returns a global default if the stack is empty."""
        ret = super(_DefaultOptsStack, self).get_default()
        return ret

    def reset(self):
        super(_DefaultOptsStack, self).reset()
        self._global_default_opts = None

_default_opts_stack = _DefaultOptsStack()

# for specific function binding
# key: function as key, value is a instance of _DefaultOptsStack
_default_opts_stack_map = defaultdict(_DefaultOptsStack)


def reset_default_opts(key=None):
    """Clears the default opts stack and resets the global default opts.

    NOTE: The default opts is a property of the current thread. This
    function applies only to the current thread.
    """
    if key is None:
        _default_opts_stack.reset()
    else:
        stack = _default_opts_stack_map.pop(key, None)
        if stack is not None:
            stack.reset()
            del stack


def default_arg(fkey, **kwargs):
    """
    example::
        tf.default_scope('conv', kernel=3, stride=1):
            x = x.conv(32)

    :param fkey: function name
    :param kwargs: default arguments for function
    :return: default context scope
    """
    if isinstance(fkey, (tuple, list)):
        return _MultiContext(Dic(**kwargs).to_default(f) for f in fkey)
    else:
        return Dic(**kwargs).to_default(fkey)


def default_args(**kwargs):
    """
    example::
        import sflow.tf as tf
        tf.default_scopes(conv=dict(kernel=3, stride=1),
                          maxpool=dict(kernel=2)):
            x = x.conv(32).relu().maxpool()

    :param kwargs:
    :return: context
    """

    # return tuple(Dic(**v).to_default(k) for k, v in kwargs.items())
    return _MultiContext(Dic(**v).to_default(k) for k, v in kwargs.items())


class _MultiContext(list):
    """ simple helper for multiple defaults arg contexts"""
    def __enter__(self, *args, **kwargs):
        for i, c in enumerate(self):
            self[i].__enter__(*args, **kwargs)
        return self

    def __exit__(self, *args, **kwargs):
        while len(self) > 0:
            self.pop().__exit__(*args, **kwargs)


def get_default_opts(key=None):
    """Returns the default opts for the current thread.

    The returned graph will be the innermost graph on which a
    `Graph.as_default()` context has been entered, or a global default
    graph if none has been explicitly created.

    NOTE: The default graph is a property of the current thread. If you
    create a new thread, and wish to use the default graph in that
    thread, you must explicitly add a `with g.as_default():` in that
    thread's function.

    Returns:
    The default `Graph` being used in the current thread.
    """
    if key is None:
        return _default_opts_stack.get_default()
    else:
        opt_stack = _default_opts_stack_map.get(key, None)
        if opt_stack is None:
            return None
        return opt_stack.get_default()


def call_with_default_context(fn, *args, **kwargs):
    opt = get_default_opts(fn.__name__)
    if opt is None:
        return fn(*args, **kwargs)
    else:
        opt = dict(opt)
        opt.update(kwargs)
        return fn(*args, **opt)


def default_deco(fn):

    @wraps(fn)
    def wrapped(*args, **kwargs):
        return call_with_default_context(fn, *args, **kwargs)

    return wrapped

