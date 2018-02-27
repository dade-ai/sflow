# -*- coding: utf-8 -*-
from __future__ import absolute_import
import sflow.core as tf
import contextlib

from collections import (OrderedDict, Mapping)

# cf : https://github.com/tensorflow/tensorflow/tree/r1.0/tensorflow/python/saved_model
# saved_model


def info_checkpoint(fpattern):
    """
    checkpoint file. variable name and shapes
    :param fpattern: checkpoint file path
    :return: {variable_saved_name: shapeinfo}
    """
    from tensorflow.python import pywrap_tensorflow

    reader = pywrap_tensorflow.NewCheckpointReader(fpattern)
    var_to_shape_map = reader.get_variable_to_shape_map()
    keys = sorted(var_to_shape_map.keys())
    d = OrderedDict()
    for k in keys:
        d[k] = (var_to_shape_map[k], reader.get_tensor(k).dtype)

    return d


def load_checkpoint(fpattern):
    """
    load checkpoint file. sorted by variable_name !!!
    :param fpattern: checkpoint file path
    :return: {variable_saved_name: value}
    """
    from tensorflow.python import pywrap_tensorflow

    reader = pywrap_tensorflow.NewCheckpointReader(fpattern)
    var_to_shape_map = reader.get_variable_to_shape_map()
    keys = sorted(var_to_shape_map.keys())
    d = OrderedDict()
    for k in keys:
        d[k] = reader.get_tensor(k)
    return d


def convert_checkpoint(inpath, savepath, name_map, relative_path=True, scope=None):
    """
    change variables mapping
    :param inpath: checkpoint input file
    :param savepath: output file
    :param relative_path:
    :param name_map: {oldname_or_var: newname_or_var} | newname = fn(oldname) | [[oldvar][newvar]]
    :param scope: the name of root scope. for `import_meta_graph` and `restore pattern`
    :return: outputpath, saver
    """
    # load old weight mapping
    weights = load_checkpoint(inpath)

    scope = scope or ''

    with tf.Graph().as_default() as g:
        with tf.variable_scope(scope):
            vars = []
            if isinstance(name_map, Mapping):
                for old, name in name_map.items():
                    vars.append(tf.Variable(initial_value=weights.pop(old), name=name))
            elif callable(name_map):
                # if functional
                for old, value in weights.items():
                    # build network for just assign value to name
                    vars.append(tf.Variable(initial_value=value, name=name_map(old)))
                weights = {}
            elif isinstance(name_map, (tuple, list)) and len(name_map) == 2:
                # tensor to new tensor
                for old, new in zip(*name_map):
                    vars.append(tf.Variable(initial_value=weights.pop(old.op.name), name=new.op.name))
            else:
                raise ValueError('check name_map type {!s}'.format(name_map))

            # remained variables
            for name, value in weights.items():
                vars.append(tf.Variable(initial_value=value, name=name))

        tf.global_variables_initializer().run()

        saver = tf.train.Saver(var_list=vars, save_relative_paths=relative_path)
        fpath = saver.save(g.session, savepath, write_meta_graph=False)

        tf.logg.info('checkpoint converted, {}->{}'.format(inpath, savepath))

    return fpath, saver


def restore_checkpoint(fpath, var_list=None, scope=None, exclude=None,
                       sess=None, verbose=False, initialize=True):
    """
    load ckpt file and assign value to variables
    :param var_list:
    :param exclude:
    :param scope:
    :param verbose:
    :param sess:
    :param fpath: checkpoint file path
    :param bool initialize : initialize uninitialzed
    :return: (not_restored_var, not_compatible_var)
    """
    info = info_checkpoint(fpath)

    var_list = var_list or tf.scope_collect(scope=scope, exclude=exclude)  # tf.global_variables()

    if verbose:
        tf.logg.info('[saved:' + '-' * 10)
        tf.logg.info('\n'.join(info.keys()))

    restore = [v for v in var_list if v.op.name in info.keys()]
    passed, notfit = [], []

    # check if restorable or not
    for v in restore:
        shape, dtype = info.pop(v.op.name)
        if v.get_shape().is_compatible_with(shape):
            # todo: find dtype check method
            # and v.dtype.is_compatible_with(dtype): tf.float32_ref not is_compatible_with float32?
            passed.append(v)
        else:
            notfit.append(v)

    # report about not_compatible_vars
    for v in notfit:
        tf.logg.warn('[{!s}] not compatible'.format(v.name))

    saver = tf.train.Saver(passed)
    sess = sess or tf.default_session()
    saver.restore(sess, fpath)

    if verbose:
        for v in passed:
            tf.logg.info('[{!s}] restored'.format(v.name))

    # report about not_restored_vars
    for name, (shape, dtype) in info.items():
        tf.logg.warn('var[{!s}]shape{!s}:{!s} ignored'.format(name, shape, dtype))

    tf.logg.info('restored from {!s}'.format(fpath))
    ignored = info.keys()

    # (not_restored_vars, not_compatible_vars)
    if initialize:
        tf.variables_initializer(tf.uninitialized_variables()).run()

    return ignored, notfit


def import_graph(fpath, input_map=None, scope=None,
                 input_colkey=None, **kwargs):
    """
    import graph from meta file. import_scope by scope or get_variable_scope()

    :param fpath: path of metafile
    :param input_map: {saved_name: tensor}, or tensors. remapping meta_graph inputs to input_tensors
    :param input_colkey: input collection key
    :param kwargs: other args for tf.train.import_meta_graph
    :param scope:
    :return: saver
    """
    # assert import_scope
    scope = scope
    if isinstance(input_map, (list, tuple)):
        input_map = {i.name: i for i in input_map}
    saver = tf.train.import_meta_graph(fpath, input_map=input_map,
                                       unbound_inputs_col_name=input_colkey,
                                       import_scope=scope,
                                       **kwargs)

    return saver


@contextlib.contextmanager
def saving_params(savepath, params, infos, scope=None,
                  key=tf.GraphKeys.TRAINABLE_VARIABLES,
                  graph=None):
    """
    assign value to variable sequentially and save checkpoint without meta
    example ::

        def _convert_vgg_params():
            params, infos = _load_lua_vgg_pretrained()
            savepath = '/train/zoo/style/adain/vgg19'

            with tf.io.saving_params(savepath, params, infos):
                x = tf.placeholder(dtype=tf.float32, shape=(None, 256, 256, 3))
                vgg19(x)

            print('done')

    :param str savepath: checkpoint path
    :param list(array) params: values to assign variables
    :param list(str) infos:
    :param str scope:
    :param tf.GraphKeys key:
    :return: None
    """
    graph = graph or tf.get_default_graph()
    with graph.as_default():
        vars0 = tf.get_collection(key, scope=scope)
        yield

        vars1 = tf.get_collection(key, scope=scope)
        vars = vars1[len(vars0):]

        assert len(params) == len(vars)

        init_ops = []
        for v, p, i in zip(vars, params, infos):
            op = v.init_op(p)
            tf.logg.info('[{}]<-[{}]'.format(v.name, i))
            init_ops.append(op)

        tf.group(*init_ops).run()
        tf.logg.info('initialized')

        # save assigned value to file
        saver = tf.train.Saver(var_list=vars)
        sess = tf.default_session()
        saver.save(sess, savepath, write_meta_graph=False)
        tf.logg.info('checkpoint saved, [{}]'.format(savepath))


def restore_graph():
    raise NotImplementedError


# def convert_graph():
#     pass


if __name__ == '__main__':
    f = '/train/zoo/vgg/vgg_16.ckpt'
    # f = '/train/zoo/vgg/vgg16.ckpt'
    # d = info_checkpoint(f)
    # d = load_checkpoint(f)

    out = '/train/zoo/vgg/vgg16_test.ckpt'
    fun = lambda x: x.replace('vgg_16', 'vgg16').replace('weights', 'W').replace('biases', 'b')
    convert_checkpoint(f, out, fun)
    d = info_checkpoint(out)
    d = load_checkpoint(out)

    print(d.keys())
