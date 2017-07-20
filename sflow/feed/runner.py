# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from contextlib import contextmanager

import threading
import tensorflow as tf
from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging
# from tensorflow.python.training import queue_runner as qr
# from tensorflow.python.training.queue_runner import QueueRunner
try:
    from tensorflow.contrib.training import FeedingQueueRunner
except ImportError:
    from tensorflow.contrib.learn.python.learn.dataframe.queues.feeding_queue_runner import FeedingQueueRunner
    pass


class GenQueueRunner(FeedingQueueRunner):

    def __init__(self, *args, **kwargs):
        self._waitempty = kwargs.pop('waitempty', False)
        super(GenQueueRunner, self).__init__(*args, **kwargs)
        # self._isempty = tf.equal(self.queue.size(), 0)

    # pylint: disable=broad-except
    def _run(self, sess, enqueue_op, feed_fn, coord=None):
        """Execute the enqueue op in a loop, close the queue in case of error.

        Args:
          sess: A `Session`.
          enqueue_op: The `Operation` to run.
          feed_fn: the feed function to pass to `sess.run`.
          coord: Optional `Coordinator` object for reporting errors and checking
            for stop conditions.

        """
        if coord:
            coord.register_thread(threading.current_thread())

        waitempty = self._waitempty
        decremented = False

        try:
            while True:
                if coord and coord.should_stop():
                    break
                try:
                    try:
                        # NOTE: @dade if generator stop wait during consuming remained data
                        feed_dict = None if feed_fn is None else feed_fn()
                        # enqueue data
                        sess.run(enqueue_op, feed_dict=feed_dict)
                    except StopIteration:
                        if coord and waitempty:
                            # wait for dequeueing
                            while not coord.should_stop():
                                # with self._lock:
                                if sess.run(self.queue.size()) == 0:
                                    raise StopIteration
                        raise StopIteration

                except (errors.OutOfRangeError, errors.CancelledError, StopIteration):
                    # This exception indicates that a queue was closed.
                    with self._lock:
                        self._runs_per_session[sess] -= 1
                        decremented = True
                        if self._runs_per_session[sess] == 0:
                            try:
                                sess.run(self._close_op)
                            except Exception as e:
                                # Intentionally ignore errors from close_op.
                                logging.vlog(1, "Ignored exception: %s", str(e))
                        return
        except Exception as e:
            # This catches all other exceptions.
            if coord:
                coord.request_stop(e)
            else:
                logging.error("Exception in QueueRunner: %s", str(e))
                with self._lock:
                    self._exceptions_raised.append(e)
                raise
        finally:
            # Make sure we account for all terminations: normal or errors.
            if not decremented:
                with self._lock:
                    self._runs_per_session[sess] -= 1


