# -*- coding: utf-8 -*-
from functools import update_wrapper
from six import wraps


class iterable(object):
    """
    decorator
    ex)
    @iterable
    def example(n):
        for i in range(n):
            yield i
    e1 = example(10)
    e2 = example(8)
    for i in e1:
        print i
    for i in e2:
        print i
    """
    def __init__(self, gen):
        self._gen = gen

    def __call__(self, *args, **kwargs):
        return iterator(self._gen, *args, **kwargs)


class iterator(object):
    """
    reiterable object
    """

    def __init__(self, gen, *args, **kwargs):
        import threading

        self._gen = gen
        self._it = None
        self._args = args  # ()
        self._kwargs = kwargs  # {}
        self._lock = threading.Lock()

        update_wrapper(self, gen)

        if args or kwargs:
            self.reset()

    def __call__(self, *args, **kwargs):
        """
        decorator call
        """
        self._args = args
        self._kwargs = kwargs
        self.reset()
        return self

    def __iter__(self):
        if self._it is None:
            self.reset()
        return self

    def next(self):
        # http://anandology.com/blog/using-iterators-and-generators/
        with self._lock:
            if self._it is None:
                raise StopIteration
            try:
                return next(self._it)
            except StopIteration:
                self._it = None
                raise StopIteration

    def __next__(self):
        # python3
        return self.next()

    def reset(self):
        self._it = self._gen(*self._args, **self._kwargs)

    def __str__(self):
        return str(self._gen)

    def __repr__(self):
        return str(self._gen)


def iterate(g, *args, **kwargs):
    it = iterator(g)
    return it(*args, **kwargs)


@iterable
def forever(it):
    """ forever
    todo : add example
    """
    while True:
        # generator 두번쨰 iteration 무한 루프 방지
        i = iter(it)
        try:
            yield i.next()
        except StopIteration:
            raise StopIteration
        while True:
            try:
                yield i.next()
            except StopIteration:
                break

