# -*- coding: utf-8 -*-
from contextlib import contextmanager
import tensorflow as tf


@contextmanager
def feeding(sess=None, coord=None):
    from .icore import default_session
    """

    :param coord: tf.train.Coordinator
    :param sess: tf.Session
    :return: coordinator
    """

    # default session
    sess = sess or default_session()
    # thread coordinator
    coord = coord or tf.train.Coordinator()

    try:
        # start queue thread
        threads = tf.train.start_queue_runners(sess, coord)
        if threads is None:
            raise ValueError('threads not created, threads == None')

        yield sess, coord

    except tf.errors.OutOfRangeError:
        print('feeding done')
    except tf.errors.CancelledError:
        print('feeding canceled')
    finally:
        # stop queue thread
        coord.request_stop()
        # wait thread to exit.

        # noinspection PyUnboundLocalVariable
        coord.join(threads)


def feeds(data):
    """
    feed the evaluated data

    example::
        for d in tf.feeds(data):
            print(d)

    :param tf.Tensor|tf.Dic|list data:
    :return:
    """
    with feeding() as (sess, coord):
        while not coord.should_stop():
            yield sess.run(data)

