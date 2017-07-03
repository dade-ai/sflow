# -*- coding: utf-8 -*-
import tensorflow as tf
from . import functional_helper as helper


def for_range(n, step, *initials, **whileopt):
    """

    example::

    pseudo(?) code ::

        def for_range(n, step, initials):
            outs = initials
            for i in xrange(n):
                outs = step(outs)
            return outs
    :param n:
    :param step:
    :param initials:
    :param whileopt: shape_invariants=None, parallel_iterations=10, back_prop=True, swap_memory=False, name=None
    :return:
    """
    i, n = tf.constant(0), tf.constant(n)

    step, initials, istuple = helper.args_or_arg(step, initials)
    flat = helper.flatpack(istuple)

    def body(t, varg):
        return flat(t + 1, step(varg))

    loopv = flat(i, initials)

    outs = tf.while_loop(helper.if_less_than(n), body, loopv, **whileopt)

    return helper.unpack_if_one(outs[1:])


def for_range_outs(n, step, *initials, **whileopt):
    """
    pseudo(?) code ::

        def for_range_outs(n, step, initial):
            outs = []
            out = initial
            for i in xrange(n):
                out = step(out)
                outs.append(out)
            return tf.pack(outs)

    :param int n: loop count
    :param Tensor(s) -> Tensor(s) step:
    :param list[Tensor] initials:
    :param whileopt: shape_invariants=None, parallel_iterations=10, back_prop=True, swap_memory=False, name=None
    :return: tf.Tensor
    """
    from snipy.basic import tupleout

    i, N = tf.constant(0), tf.constant(n)
    step = tupleout(step)
    initials = map(tf.identity, initials)

    outs_ta = map(lambda x: tf.TensorArray(x.dtype, N), initials)
    narg = len(initials)

    def body(t, *args):
        res = step(*args[:narg])
        outs = list(args[narg:])
        for i, r in enumerate(res):
            outs[i] = outs[i].write(t, r)
        return [t + 1] + list(res) + list(outs)

    loopv = [i] + list(initials) + outs_ta
    outs = tf.while_loop(helper.if_less_than(N), body, loopv, **whileopt)
    outs = outs[1:]
    lastout = outs[:narg]
    outs_ta = outs[narg:]

    for i, last in enumerate(lastout):
        outs_ta[i] = outs_ta[i].stack()
        outs_ta[i].set_shape([n] + last.get_shape().dims)

    return helper.unpack_if_one(outs_ta)


@helper.functional_factory(left=True)
def foldleft(step, seq, istate=None, name=None, **whileopt):
    """
    scan 단, 마지막 값만 넘겨준다. ::
        def _for_seq(step, seq, istate):

            n = tf.shape(seq)[0]
            state = istate
            for iseq in seq.split(0, n):
                state = step(iseq, state)

            return state

    :param (seq, state) -> state step:
    :param Tensor seq:
    :param Tensor istate:
    :param whileopt:
    :return: Tensor laststate
    """
    pass


@helper.functional_factory(left=False)
def foldright(step, seq, istate=None, name=None, **whileopt):
    # todo : add docstring
    pass


@helper.functional_factory(left=False, flipout=True)
def foldrev(step, seq, istate=None, name=None, **whileopt):
    # todo : add docstring
    pass


@helper.functional_factory(left=True, condfn=True, flipout=False)
def foldleft_if(cond, step, seq, istate=None, name=None, **whileopt):
    # todo : add docstring
    pass


@helper.functional_factory(left=False, condfn=True, flipout=False)
def foldright_if(cond, step, seq, istate=None, name=None, **whileopt):
    # todo : add docstring
    pass


@helper.functional_factory(left=False, condfn=True, flipout=True)
def foldrev_if(cond, step, seq, istate=None, name=None, **whileopt):
    # todo : add docstring
    pass


@helper.functional_factory(left=True, accum=True)
def scanleft(step, seq, istate=None, name=None, **whileopt):
    """
    tf.scan과 거의 동일. tf.scan shape issue fix 아래는 대략의 로직::
        def _for_seq_outs(step, seq, istate):
            states = []
            state = istate
            n = seq.get_shape()[0].value
            for iseq in seq.split(0, n):
                state = step(state, iseq)
                states.append(state)  # 벡터면 squeeze() 한 것 처럼 된다.

            return tf.stack(states)

    :param (seq_t, state_t1) -> state_t step:
    :param Tensor seq: sequence tensor, 0 axis가 time. batch dim은 바깥 루프에서 해결할 것
    :param Tensor istate: 초기 state
    :param name:
    :param kwargs:
    :return:
    """
    pass


@helper.functional_factory(left=False, accum=True)
def scanright(step, seq, istate=None, name=None, **whileopt):
    # todo : add docstring
    pass


@helper.functional_factory(left=False, accum=True, flipout=True)
def scanrev(step, seq, istate=None, name=None, **whileopt):
    # todo : add docstring
    pass


@helper.functional_factory(left=True, condfn=True, accum=True, flipout=False)
def scanleft_if(cond, step, seq, istate=None, name=None, **whileopt):
    # todo : add docstring
    pass


@helper.functional_factory(left=False, condfn=True, accum=True, flipout=False)
def scanright_if(cond, step, seq, istate=None, name=None, **whileopt):
    # todo : add docstring
    pass


@helper.functional_factory(left=False, condfn=True, accum=True, flipout=True)
def scanrev_if(cond, step, seq, istate=None, name=None, **whileopt):
    # todo : add docstring
    pass


# todo : add example code?
