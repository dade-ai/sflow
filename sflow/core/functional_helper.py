# -*- coding: utf-8 -*-
# region utils
import functools

import tensorflow as tf
from tensorflow.python.framework import ops


def functional_factory(left=True, condfn=None, accum=False, flipout=False):

    # todo : add docstring and explanations

    def deco(fn):

        def get_body(step, seq, inc, n, outs=None):

            def body(t, state_t):
                return t + inc, step(seq.read(t), state_t)

            def body_accum(t, state_t, outs):
                out = step(seq.read(t), state_t)
                outs = outs.write(t, out)
                return t + inc, out, outs

            def body_accum_flip(t, state_t, outs):
                out = step(seq.read(t), state_t)
                outs = outs.write(n - 1 - t, out)
                return t + inc, out, outs

            return body if not outs else (body_accum_flip if flipout else body_accum)

        @functools.wraps(fn)
        def wrapped(step, seq, istate=None, cond=condfn, name=None, **whileopt):
            with tf.variable_scope(name, fn.__name__, [seq, istate]) as varscope, \
                    ops.colocate_with(seq):
                # Any get_variable calls fn will cache the first call locally
                # and not issue repeated network I/O requests for each iteration.
                if varscope.caching_device is None:
                    varscope.set_caching_device(lambda op: op.device)

                return _step_out_only(step, seq, istate, cond, **whileopt)

        @functools.wraps(fn)
        def wrapped_if(cond, step, seq, istate=None, name=None, **whileopt):
            return wrapped(step, seq, istate=istate, cond=cond, name=name, **whileopt)

        def _step_out_only(step, seq, istate, cond, **whileopt):
            n = tf.shape(seq)[0]
            shape = seq.shape
            seq = tensor_array(seq, size=n)
            noinit = (istate is None)

            start = 0 if left else n - 1
            inc = 1 if left else -1
            if noinit:
                istate = seq.read(start)
                # istate.set_shape(seq.shape[1:])
                istate.set_shape(shape[1:])
                t = tf.constant(start + inc)
            else:
                t = tf.convert_to_tensor(start)

            if accum:
                outs = tf.TensorArray(dtype=istate.dtype, size=n, clear_after_read=True)
                if noinit:
                    iw = start if not flipout else n - start + 1
                    outs = outs.write(iw, istate)
            else:
                outs = None

            no_cond = (cond is None)
            cond = cond or if_less_than(n) if left else if_greater_equal(0)
            body = get_body(step, seq, inc, n, outs)

            if accum:
                t, lastout, outs = tf.while_loop(cond, body, (t, istate, outs), **whileopt)
                outs = outs.stack()
                # todo : check this. dade
                n = shape[0].value

                size = n if no_cond else (t if left else n-t)
                outs.set_shape([size] + lastout.get_shape().dims)
            else:
                _, outs = tf.while_loop(cond, body, (t, istate), **whileopt)

            return outs

        return wrapped if not condfn else wrapped_if

    return deco


def args_or_arg(fn, args):
    # todo : add docstring
    isargs = len(args) > 1

    def fnargs(outin):
        return fn(*outin)

    if isargs:
        return fnargs, args, isargs
    else:
        return fn, args[0], isargs


def flatpack(isargs):
    from snipy.basic import tuple_pack
    # todo : add docstring

    def return_arg(i, arg):
        return i, arg

    def return_args(i, args):
        return tuple_pack(i, *args)

    return return_args if isargs else return_arg


def unpack_if_one(atuple):
    if len(atuple) == 1:
        return atuple[0]
    else:
        return atuple


def tensor_array(x, size, **kwargs):
    ta = tf.TensorArray(x.dtype, size=size, **kwargs)
    # return ta.unpack(x) : todo : check api changed?
    return ta.unstack(x)

# endregion

# region conditions


def if_less_than(n):

    def _if_less_than(i, *_):
        return i < n
    return _if_less_than


def if_greater_equal(n):
    def _if_greater_equal(i, *_):
        return i >= n
    return _if_greater_equal


# endregion
