# -*- coding: utf-8 -*-
from __future__ import absolute_import
from snipy.activeq import ActiveQ
from snipy.queues import (QueueInterruptable, is_main_alive)
from .iflag import flag
# from snipy.cuda import get_cuda_device_count
import os
from time import sleep
from snipy.ilogging import logg


def _iter_combination(args, kwargs):
    # get all parameter combinations
    from itertools import product
    nargs = len(args)

    keys = kwargs.keys()
    values = kwargs.values()
    params = list(args) + values
    for comb in product(*params):
        cargs = comb[:nargs]
        cvalue = comb[nargs:]
        kws = dict(zip(keys, cvalue))

        yield cargs, kws


def _run_gpu(cuda, script, *args, **kwargs):
    import sys

    params = ['--{}={}'.format(k, v) for k,v in kwargs.items()]
    cmd = 'python {} {} {} --cuda={} '.format(script, ' '.join(args), ' '.join(params), cuda)

    logg.info('cmd: [{}]'.format(cmd))

    res = os.system(cmd)
    sys.stdout.flush()

    return res


def _run_gpu_test(cuda, script, *args, **kwargs):
    import sys

    params = ['--{}={}'.format(k, v) for k,v in kwargs.items()]
    cmd = 'python {} {} {} --cuda={} '.format(script, ' '.join(args), ' '.join(params), cuda)

    logg.info('cmd: [{}]'.format(cmd))

    # res = os.system(cmd)
    sys.stdout.flush()

    return 0


class CudaJobQ(ActiveQ):

    def __init__(self, jobq, cuda, script, action=None):
        self.cuda = cuda
        self.script = script
        self._action = action or _run_gpu
        super(CudaJobQ, self).__init__(maxsize=10, jobq=jobq)

    def action(self, item):
        return self._action(self.cuda, self.script, *item[0], **item[1])

    def join(self):
        if self._run_thread:
            self._run_thread.join()


def get_cuda_nums():
    from snipy.cuda import get_cuda_device_count
    logg.info('flag.cuda: {}'.format(flag.cuda))
    if not flag.cuda:
        # resotre me
        count = get_cuda_device_count()
        # count = 8
        flag.cuda = ','.join(str(i) for i in range(count))

    return flag.cuda.split(',')


def _try_as_list(param_str):
    """

    :param str param_str:
    :return:
    """
    if not param_str.startswith('['):
        return [param_str]
    else:
        params = eval(param_str)
        assert isinstance(params, list)
        return params


def run_grid(script, *args, **kwargs):
    """
    run script with each combination of parameters
    :param str script: python script
    :param args:
    :param kwargs:
    :return:
    """
    from snipy.iterflow import iqueue
    script = script[0]

    test = kwargs.pop('test', False)
    action = _run_gpu_test if test else None

    logg.info('script == {}'.format(script))
    logg.info('args== {}'.format(str(args)))

    logg.info('flag.cuda: {}'.format(flag.cuda))
    # flag.cuda = kwargs.pop('cuda', None)

    args = list(map(_try_as_list, args))
    logg.info('args== {}'.format(str(args)))
    kwargs = {k: _try_as_list(v) for k, v in kwargs.items()}

    # all parameter combinations
    jobq = iqueue(_iter_combination(args, kwargs))

    # cudajob_q foreach gpu
    cudaq = [CudaJobQ(jobq, str(i), script, action=action).start() for i in get_cuda_nums()]

    while not jobq.empty():
        sleep(1)

    # wait for all job finished
    for q in cudaq:
        while not q.empty():
            sleep(0.5)
        q.stop()
        q.join()

    while not jobq.done:
        continue

    logg.info('done!')
    return 0


if __name__ == '__main__':
    # ex) python -m sflow.tf.rungrid script --cuda=0,1,2  --param1=[1,2,3] --param2=[2,3,4]
    import sflow.tf as tf
    tf.run(run_grid)
