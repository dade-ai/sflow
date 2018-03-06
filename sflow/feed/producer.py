# -*- coding: utf-8 -*-
from __future__ import absolute_import
import tensorflow as tf
from .runner import GenQueueRunner
from .reader import read_image


def _feed_fn(gen, placeholders):

    def feed_fn_one():
        data = next(gen)
        return {placeholders[0]: data}

    def feed_fn():
        data = next(gen)

        # # debug temp
        # if placeholders[0].dims[2] != data[0].shape[2]:
        #     temp_debug_me = 1

        return {p: d for p, d in zip(placeholders, data)}

    if not isinstance(placeholders, (tuple, list)):
        placeholders = [placeholders]
    if len(placeholders) == 1:
        return feed_fn_one
    return feed_fn


def gen_producer(placeholders, gen, capacity, min_after_dequeue=0,
                 threads=1, shuffle=False, enqueue_many=False, summary_name=None):
    """
    example::
        todo : add some example

    :param placeholders:
    :param gen:
    :param capacity:
    :param min_after_dequeue:
    :param threads:
    :param shuffle:
    :param enqueue_many:
    :return:
    """
    from ..core.logg import logg

    if not isinstance(placeholders, (tuple, list)):
        placeholders = [placeholders]

    dtypes = [p.dtype for p in placeholders]
    if any(not p.shape.is_fully_defined() for p in placeholders):
        shapes = None
        # warning.. no dequeue many
    else:
        shapes = [p.get_shape() for p in placeholders]

    if shuffle:
        # q = tf.RandomShuffleQueue(capacity, min_after_dequeue, dtypes, shapes=shapes, names=names)
        q = tf.RandomShuffleQueue(capacity, min_after_dequeue, dtypes, shapes=shapes)
        waitempty = (min_after_dequeue == 0)
    else:
        # q = tf.FIFOQueue(capacity, dtypes, shapes=shapes, names=names)
        q = tf.FIFOQueue(capacity, dtypes, shapes=shapes)
        waitempty = True
    # feeddic = {n: p for n, p in zip(names, placeholders)}

    names = [p.name for p in placeholders]
    logg.info(str(names))

    if enqueue_many:
        enq = q.enqueue_many(placeholders)
    else:
        enq = q.enqueue(placeholders)

    if summary_name is not None:
        tf.summary.scalar(summary_name, tf.cast(q.size(), tf.dtypes.float32) * (1. / capacity))

    feedgen = _feed_fn(gen, placeholders)
    qr = GenQueueRunner(queue=q, enqueue_ops=[enq]*threads, feed_fns=[feedgen]*threads,
                        waitempty=waitempty)
    tf.train.add_queue_runner(qr)

    return q


def queue_producer(tensors, capacity, shapes=None, threads=1, shuffle=False, min_after_dequeue=0,
                   enqueue_many=False):
    """
    todo : add example
    :param tensors:
    :param capacity:
    :param shapes:
    :param threads:
    :param shuffle:
    :param min_after_dequeue:
    :param enqueue_many:
    :return:
    """
    if not isinstance(tensors, (tuple, list)):
        tensors = [tensors]
    dtypes = [img.dtype for img in tensors]
    shapes = shapes or [img.get_shape() for img in tensors]
    if shuffle:
        q = tf.RandomShuffleQueue(capacity, min_after_dequeue, dtypes=dtypes, shapes=shapes)
        # waitempty = (min_after_dequeue == 0)
    else:
        q = tf.FIFOQueue(capacity, dtypes=dtypes, shapes=shapes)
        # waitempty = True
    if enqueue_many:
        enq = q.enqueue_many(tensors)
    else:
        enq = q.enqueue(tensors)

    qr = tf.train.QueueRunner(q, enqueue_ops=[enq]*threads)
    tf.train.add_queue_runner(qr)

    return q


def image_read_producer(fname, capacity, shape=None, preprocess=None, threads=1,
                        shuffle=False, min_after_dequeue=0, channels=None):
    """
    example::
        todo : add some example

    :param fname:
    :param capacity:
    :param shape:
    :param preprocess:
    :param threads:
    :param shuffle:
    :param min_after_dequeue:
    :param channels:
    :return:
    """
    img = read_image(fname, channels=channels)

    if preprocess is not None:
        img = preprocess(img)

    q = queue_producer(img, capacity, shapes=[shape], threads=threads,
                       shuffle=shuffle, min_after_dequeue=min_after_dequeue)
    return q


def read_matching_image(pattern, **kwargs):
    """

    :param pattern:
    :param kwargs:
    :return:
    """
    files = tf.matching_files(pattern)
    q = image_read_producer(files, **kwargs)

    return q
