# -*- coding: utf-8 -*-
# from __future__ import absolute_import
from contextlib import contextmanager
import collections
import tensorflow as tf

from .iconst import const
from .defaults import Dic
from snipy.basic import (patchmethod, patchproperty, tuple_args)


def get_var(name, shape=None, dtype=None, initializer=None, **kwargs):
    """
    variable create and init
    :param value:
    :param Callable initializer:
    :param shape:
    :param dtype:
    :param name:
    :param kwargs:
    :return:
    """
    def _common_dtype(value, d):
        if d is not None:
            return d
        try:
            if value.dtype == 'float64':
                d = const.floatx
            elif value.dtype == 'int64':
                d = const.intx
        except AttributeError:
            d = value.dtype
        return d

    if not callable(initializer):
        # If initializer is a constant or tensor, do not specify shape.
        initializer = tf.convert_to_tensor(initializer, dtype=dtype)
        assert initializer.shape == shape
        dtype = _common_dtype(initializer, dtype)
        v = tf.get_variable(name=name, dtype=dtype, initializer=initializer, **kwargs)
    else:
        v = tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=initializer, **kwargs)

    return v


def default_session(graph=None):
    g = graph or tf.get_default_graph()
    return g.session


# region lazy graph property

@patchproperty(tf.Graph, property='session')
def _graph_session(g):
    try:
        s = g._session
    except AttributeError:
        s = session(g)
        g._session = s
    return s


@patchproperty(tf.Graph, property='root_scope')
def assert_root_scope(g=None):
    g = g or tf.get_default_graph()
    try:
        sc = g._root_scope
    except AttributeError:
        with tf.variable_scope('') as sc:
            g._root_scope = sc
    return sc


assert_root_scope(tf.get_default_graph())


# @patchproperty(tf.Graph, property='seed')
def get_graph_seed(g=None):
    """
    enforce random seed to be determinstic
    :param g:
    :return:
    """
    g = g or tf.get_default_graph()
    if g._seed is None:
        g._seed = 1004
    return g._seed


def set_graph_seed(g, value):
    """
    enforce random seed to be determinstic
    :param g:
    :return:
    """
    g = g or tf.get_default_graph()
    g._seed = value

tf.Graph.seed = property(get_graph_seed, set_graph_seed)


def graph_scope(g=None):
    g = g or tf.get_default_graph()
    return g.root_scope


def Graph():
    g = tf.Graph()
    # noinspection PyStatementEffect
    assert_root_scope(g)

    return g


@patchproperty(tf.Graph)
def is_training(graph=None):
    g = graph or tf.get_default_graph()
    try:
        v = g._is_training
    except AttributeError:
        with tf.variable_scope(g.root_scope):
            v = get_var('is_training', initializer=True, dtype=tf.bool, shape=(), trainable=False)
        g._is_training = v
    return v


@patchproperty(tf.Graph)
def global_step(graph=None):
    g = graph or tf.get_default_graph()
    try:
        v = g._global_step
    except AttributeError:
        with tf.variable_scope(g.root_scope):
            v = get_var('global_step', initializer=0, dtype=tf.int64, trainable=False, shape=(),
                        collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP],
                        )
        g._global_step = v
    return v

# endregion


@patchmethod(tf.Variable)
def init_op(v, value):
    init = v._initializer_op = v.assign(value).op
    return init


@patchmethod(tf.Variable)
def init_value(v, value):
    # init = v._initializer_op = v.assign(value).op
    # v.session.run(init)
    v.init_op(value).run()


def set_training(training, graph=None):
    g = graph or tf.get_default_graph()
    training = True if training else False
    g.is_training.init_value(training)


def get_global_step(graph=None):
    return global_step(graph)


def set_global_step(ep, graph=None):
    # g = graph or tf.get_default_graph()
    # g.global_step.init_value(ep)
    # return g.global_step
    v = get_global_step(graph)
    v.init_value(ep)
    return v


# def get_tensors_by_scope(graph, scope):
#     pass

# endregion


# region session related patch


def session(graph=None, config=None):
    """
    https://www.tensorflow.org/how_tos/using_gpu/
    :param graph:
    :param config:
    :return:
    """
    if config is None:
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.7

        # config = tf.ConfigProto(allow_soft_placement=False)
        # config.gpu_options.allocator_type = 'BFC'
        # config.gpu_options.allow_growth = False
    sess = tf.Session(config=config, graph=graph)
    # sess.graph = graph
    return sess  #.to_default()


@patchproperty(tf.Tensor, tf.Variable, tf.Operation, property='session')
def _session_property(t):
    return t.graph.session


tf.Session.__base__._org_run = tf.Session.__base__.run


@patchmethod(tf.Session.__base__, name='run')
def _run_session(sess, fetches, feed_dict=None, options=None, run_metadata=None, **kwargs):
    fetches = Dic.dic_to_dict(fetches)
    out = sess._org_run(fetches, feed_dict=feed_dict, options=options, run_metadata=run_metadata, **kwargs)
    return Dic.dict_to_dic(out)


# endregion


# region variable wrappers

def variable(initializer=None, shape=None, dtype=None, name=None, **kwargs):
    """
    variable create and init
    :param initializer: callable(shape) or value
    :param shape:
    :param dtype:
    :param name:
    :param kwargs:
    :return:
    """
    return get_var(name, shape=shape, dtype=dtype, initializer=initializer, **kwargs)


def get_weight(name, **kwargs):
    c = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS]
    return get_var(name, collections=c, **kwargs)


def get_bias(name, **kwargs):
    c = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES]
    return get_var(name, collections=c, **kwargs)


# endregion


# region eval and value related


@patchmethod(tf.Variable)
def set_value(t, value):
    t.session.run(t.assign(value))
    # t.session.run(tf.assign(t, value))


tf.Variable._eval = tf.Variable.eval
tf.Tensor._eval = tf.Tensor.eval


# noinspection PyShadowingNames
@patchmethod(tf.Tensor, name='eval')
def _tensor_eval(t, feed_dict=None, session=None):
    # noinspection PyProtectedMember
    return t._eval(feed_dict=feed_dict, session=session or t.session)  # tf.get_default_session())  # ,t.session)


# noinspection PyShadowingNames
@patchmethod(tf.Variable, name='eval')
def _variable_eval(v, session=None):
    # noinspection PyProtectedMember
    return v._eval(session=session or v.session)  # tf.get_default_session()) #v.session)

tf.Operation._run = tf.Operation.run


# noinspection PyShadowingNames
@patchmethod(tf.Operation, name='run')
def _run(o, feed_dict=None, sess=None):
    # noinspection PyProtectedMember
    return o._run(feed_dict=feed_dict, session=sess or o.session)  # tf.get_default_session())


def initialize_if_not(sess=None):
    tf.variables_initializer(uninitialized_variables(sess=sess)).run()


@tuple_args
def restore_or_initialize(savers, ep=None):
    for saver in savers:
        saver.restore(ep=ep)

    # check uninitialized variables
    tf.variables_initializer(uninitialized_variables()).run()


def uninitialized_variables(sess=None, verbose=True):
    varlist = tf.global_variables() + tf.model_variables()

    try:
        # for old version
        uninitialized = tf.concat_v2(tf.logical_not([tf.is_variable_initialized(v) for v in varlist]), 0)
    except AttributeError:
        uninitialized = tf.concat(values=tf.logical_not([tf.is_variable_initialized(v) for v in varlist]), axis=0)

    mask = uninitialized.eval(session=sess)
    # gather?
    unvars = [v for v, m in zip(varlist, mask) if m]
    return unvars


def scope_collect(scope=None, exclude=None, key=None):
    """
    :param str scope: scope string pattern
    :param exclude: exclude
    :param key: key of collection
    :return:
    """
    if scope is not None or exclude is not None:
        scope = scope or ''
        scope = scope_re(scope, exclude=exclude)

    key = key or tf.GraphKeys.GLOBAL_VARIABLES
    return tf.get_collection(key, scope)


# region function binding

def _list_tensor_dict(d):
    """
    :param dict d:
    :return:
    """
    info = Dic()
    tensors = list()
    for k, v in d.items():
        if isinstance(v, (tuple, list)):
            tensors.extend(v)
            info[k] = len(v)
        elif isinstance(v, dict):
            l, info = _list_tensor_dict(v)
            tensors.extend(l)
            info[k] = info
        else:  # Tensor
            tensors.append(v)
            info[k] = 1
    return tensors, info


def _restore_tensor_dict(l, info):
    """
    :param list l:
    :param dict info:
    :return:
    """
    def restore(infodic, i):
        d = Dic()
        for k, v in infodic.items():
            if isinstance(v, dict):
                d[k], i = restore(v, i)
            elif v == 1:
                d[k] = l[i]
                i += 1
            else:
                d[k] = l[i:i+v]
                i += v
        return d, i

    d, _ = restore(info, 0)
    return d


def function(inputs, outputs, sess=None, preprocess=None, postprocess=None):
    """
    theano-like interface
    example::

    :param list[PlaceHolder] inputs:
    :param list[Tensor] | dic[str, Tensor] outputs:
    :return: function
    """

    def asis(x):
        return x

    preprocess = preprocess if preprocess else asis
    postprocess = postprocess if postprocess else asis

    if isinstance(outputs, collections.Mapping):  # dict out
        outputs, infodict = _list_tensor_dict(outputs)
        form_output = lambda out: _restore_tensor_dict(out, infodict)
    else:
        form_output = asis

    if isinstance(inputs, (tuple, list)):
        narg = len(inputs)
    else:
        narg = 0  # not list or tuple
        inputs = [inputs]
    assert form_output

    def call_simple(*feedvalues, **kwargs):
        if len(feedvalues) == 1 and narg > 0:
            feedvalues = feedvalues[0]

        feedvalues = preprocess(feedvalues)

        try:
            assert len(feedvalues) == (narg or 1)
        except AssertionError as e:
            raise e

        feed_dict = kwargs.pop('feed_dict', dict())
        feed_dict.update(zip(inputs, feedvalues))

        # fixme: weakref problem
        sess_ = sess or default_session()
        outs = sess_.run(outputs, feed_dict=feed_dict)

        res = form_output(outs)
        return postprocess(res)

    return call_simple


# region get variables filtered by name


@tuple_args
def scope_re(scopes, exclude=None):
    """
    make common regular expression to use with re.match (in get_collection()))
    example:
        tf.scope_re(['generator', 'discriminater'], exclude='vgg19')

    :param scopes: include patterns
    :param exclude: exclude patterns
    :return: pattern string of regular expression
    """
    patterns = ''
    if exclude is not None:
        if not isinstance(exclude, (tuple, list)):
            exclude = [exclude]
        # add exclude pattern
        patterns = '(?!{})'.format('|'.join(exclude))
    patterns += '|'.join(scopes)

    return patterns


# endregion


# region etc
_placeholder = tf.placeholder


def placeholder(shape=None, dtype=None, name=None):
    if isinstance(shape, (tf.DType, str)):
        return tf.placeholder(dtype=shape, shape=dtype, name=name)
    try:
        if isinstance(shape, (tf.DType, basestring)):
            # swap args
            return tf.placeholder(dtype=shape, shape=dtype, name=name)
    except NameError:
        # error by basestring
        pass
    dtype = dtype or tf.float32
    return tf.placeholder(dtype=dtype, shape=shape, name=name)


# endregion
