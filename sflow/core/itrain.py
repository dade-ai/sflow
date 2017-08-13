# -*- coding: utf-8 -*-
from __future__ import print_function

import tensorflow as tf

from snipy.basic import (interrupt_guard)
from . import logg
from .icore import (default_session, get_global_step)


class _SaverWrap(object):

    def __init__(self, savepath, global_step=None, epochper=1,
                 var_list=None, scope=None, keep_optim=False, **kwargs):
        """
        todo : add comment
        :param savepath:
        :param global_step:
        :param epochper:
        :param var_list:
        :param scope:
        :param kwargs:
        """
        import os

        self._global_step = global_step if global_step is not None else get_global_step()
        if scope is not None:
            if var_list is not None:
                raise ValueError('var_list and scope are mutually exclusive arg')
            # var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            # var_list += tf.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, scope=scope)
            # var_list += tf.get_collection(tf.GraphKeys.COND_CONTEXT, scope=scope)
            # var_list += [self._global_step]

            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
            var_list += tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope=scope)

            # # todo: test this
            # if not keep_optim:
            #     opt_vars = tf.get_collection('OPTIMIZER_VAR')
            #     var_list = [v for v in var_list not in opt_vars]

            if self._global_step not in var_list:
                var_list += [self._global_step]
        self.scope = scope
        assert savepath

        folder, pre = os.path.split(savepath)
        if not pre:
            pre = os.path.basename(folder)
        savepath = os.path.join(folder, pre)
        self._savepath = savepath
        self.savedir = folder
        # self._savedir = os.path.dirname(savepath)
        # = os.path.join(savepath, scope or 'ckpt')
        # self._saveper = saveper
        self._epochper = epochper
        self._saver = tf.train.Saver(var_list=var_list, **kwargs)
        self.var_list = var_list

    def restore(self, sess=None, ep=None):
        from sflow.io.save_restore_util import restore_checkpoint

        # restore or not
        if ep is not None:
            lastfile = self._savepath + '-{:d}'.format(ep)
        else:
            lastfile = tf.train.latest_checkpoint(self.savedir)

        if lastfile is not None:
            # tf.global_variables_initializer().run()
            sess = sess or default_session()

            # restore var_list without error though not found var exists
            restore_checkpoint(lastfile, var_list=self.var_list, sess=sess)
            # self._saver.restore(sess, lastfile)

            logg.info('restored from [{}]'.format(lastfile))
            return True
        else:
            logg.warn('No file to restore with path[{}]'.format(self.savedir))
        return False

    def save(self, sess, ep=None, gstep=None):
        # todo : option of ignoring optimizer variables
        gstep = gstep or self._global_step.eval()
        ep = ep or (gstep // self._epochper)

        sess = sess or default_session()
        return self._saver.save(sess, self._savepath,
                                global_step=ep,
                                write_meta_graph=False)

    def save_meta(self, sess):
        sess = sess or default_session()
        return self._saver.save(sess, save_path=self._savepath, global_step=self._global_step,
                                write_meta_graph=True)


class _SummaryWriter(object):

    def __init__(self, logdir=None, summaryop=None,
                 graph=None, max_queue=10, flush_secs=20, **kwargs):
        logdir = logdir or 'train/log'
        if not logdir.endswith('/'):
            logdir += '/'
        if graph is None:
            graph = tf.get_default_graph()
        self.logdir = logdir
        self._writer = tf.summary.FileWriter(logdir, graph=graph, max_queue=max_queue,
                                             flush_secs=flush_secs, **kwargs)
        # self.summaryper = summaryper
        # self._global_step = global_step or tf.get_global_step()
        self.summaryop = summaryop if summaryop is not None else tf.summary.merge_all()
        self._session = default_session()

    def add_summary(self, gstep, sess=None, summary=None, other_op=None):
        if summary is None:
            sess = sess or self._session
            if other_op is not None:
                summary = sess.run([self.summaryop] + other_op)[0]
            else:
                summary = sess.run(self.summaryop)

        print('[{}]'.format(self.logdir)),
        print('summary', end='')

        if isinstance(summary, (tuple, list)):
            [self._writer.add_summary(s, global_step=gstep) for s in summary]
        else:
            self._writer.add_summary(summary, global_step=gstep)


def summary_writer(logdir=None, summaryop=None, **kwargs):
    # todo add some comment
    return _SummaryWriter(logdir, summaryop, **kwargs)


def saver(savedir='train', global_step=None, epochper=1,
          var_list=None, scope=None, keep_optim=False, **kwargs):
    # todo add some comment

    return _SaverWrap(savepath=savedir, global_step=global_step,
                      epochper=epochper, var_list=var_list, scope=scope,
                      keep_optim=keep_optim,
                      **kwargs)


def init_operations(scope=None):
    ops = tf.get_collection(tf.GraphKeys.INIT_OP, scope=scope)
    return tf.group(*ops)


def backup_train_script_to(savedir, depth=2):
    from snipy.caller import caller
    from snipy.io.fileutil import (filecopy, mkdir_if_not)
    from time import time
    from sflow.tf import flag
    import os

    # get script file and copy
    forg = caller.abspath(depth)
    name = os.path.basename(forg)

    fname = os.path.join(savedir, 'backup.{}.{}'.format(str(time()), name))
    mkdir_if_not(fname)
    filecopy(forg, fname)

    # save flag values
    values = '# flag values when running : {}'.format(dict(flag))
    with open(fname, 'a') as f:
        f.write(values)


def trainall(outputs, savers=None, ep=None, maxep=None, epochper=10000, saveper=1):
    """
    todo : add example
    :param outputs:
    :param savers:
    :param ep:
    :param maxep:
    :param epochper:
    :param saveper:
    :return:
    """
    # generator step training
    from .ioptimizer import trains_increase_step
    from .icore import (restore_or_initialize, set_training, get_global_step)
    from .ifeeding import feeding

    epcount = ep

    trainop = trains_increase_step()
    global_step = get_global_step()

    if savers is None:
        savers = [saver(epochper=epochper)]
    if not isinstance(savers, (tuple, list)):
        savers = [savers]

    # leave backup script
    backup_train_script_to(savers[0].savedir)

    restore_or_initialize(savers)
    set_training(True)
    if not isinstance(outputs, (tuple, list)):
        outputs = [outputs]
    runops = list(outputs) + [global_step, trainop]

    init_operations().run()

    with feeding() as (sess, coord):
        gstep = global_step.eval()
        ep = gstep // epochper
        ep_p = ep
        if maxep is None:
            maxep = ep + (epcount or 10)
        _save_metas(savers, sess)
        while not coord.should_stop() and ep < maxep:
            try:
                outs = sess.run(runops)
                losses = outs[:-2]
                gstep = outs[-2]

                ep = gstep // epochper
                if saveper and ep_p < ep and ep % saveper == 0:
                    _save_weight(savers, sess, ep)
                yield ep, gstep, losses
            except KeyboardInterrupt:
                reraise = _save_or_not(savers, sess, ep)
                if reraise:
                    raise KeyboardInterrupt
            finally:
                ep_p = ep

        if ep == maxep:
            # save
            _save_weight(savers, sess, ep)


def trainstep(ep=None, maxep=None, epochper=1, saveper=1, savers=None, ep_restore=None,
              backup=True):
    """
    routine for alternative training
    :param ep: ep count
    :param maxep:
    :param epochper:
    :param saveper:
    :param savers:
    :param ep_restore:
    :return:
    """

    # generator step training
    from .ioptimizer import increase_step
    from .icore import (restore_or_initialize, set_training)
    from .ifeeding import feeding
    epcount = ep

    igstep = increase_step()
    global_step = get_global_step()

    if savers is None:
        savers = [saver(epochper=epochper)]
    elif not isinstance(savers, (tuple, list)):
        savers = [savers]

    # leave backup script
    if backup:
        backup_train_script_to(savers[0].savedir)

    restore_or_initialize(savers, ep=ep_restore)
    set_training(True)

    init_operations().run()

    # try:
    with feeding() as (sess, coord):
        gstep = global_step.eval()

        ep = gstep // epochper
        ep_p = ep
        if maxep is None:
            maxep = ep + (epcount or 10)

        while not coord.should_stop() and ep < maxep:
            yield ep, gstep

            try:
                gstep = sess.run([igstep, global_step])[1]
                ep = gstep // epochper

                if saveper and ep_p < ep and ep % saveper == 0:
                    _save_weight(savers, sess, ep)
            except KeyboardInterrupt:
                save_on_interrupt(savers, sess, ep)
            finally:
                ep_p = ep

        if ep == maxep:
            # save
            _save_weight(savers, sess, ep)


def _save_metas(savers, sess):
    from datetime import datetime

    if not isinstance(savers, (tuple, list)):
        savers = [savers]
    with interrupt_guard('saving.'):
        for s in savers:
            s.save_meta(sess=sess)
            print(str(datetime.now()))


def _save_weight(savers, sess, ep):
    from datetime import datetime

    if not isinstance(savers, (tuple, list)):
        savers = [savers]
    with interrupt_guard('saving.'):
        for s in savers:
            s.save(sess=sess, ep=ep)
            print(str(datetime.now()))


def save_on_interrupt(savers, sess, ep):
    reraise = _save_or_not(savers, sess, ep)
    if reraise:
        raise KeyboardInterrupt


def _save_or_not(savers, sess, ep):
    if not isinstance(savers, (tuple, list)):
        savers = [savers]
    yes = 'yes'
    try:
        yes = raw_input('\nSave Weight? [y/n] return=y')
    except NameError:
        yes = input('\nSave Weight? [y/n] return=y')
    finally:
        yes = yes.lower() not in ['n', 'no']
    if yes:
        _save_weight(savers, sess, ep)
        print('saved')
        # dont' reraise
        return False
        # reraise or not
    else:
        # reraise
        return True

