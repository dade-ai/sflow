# -*- coding: utf-8 -*-
from .datasets import dataset_reader
import sflow.tf as tf


def _folder():
    return 'ukiyoe2photo'


def dataset_train(batch, **kwargs):
    a = dataset_trainA(batch, **kwargs)
    b = dataset_trainB(batch, **kwargs)
    return tf.dic(A=a, B=b, batch=batch)


def dataset_test(batch, **kwargs):
    a = dataset_trainA(batch, **kwargs)
    b = dataset_trainB(batch, **kwargs)
    return tf.dic(A=a, B=b, batch=batch)


def dataset_trainA(batch, **kwargs):
    return dataset_reader(batch, _folder(), 'trainA', **kwargs)


def dataset_trainB(batch, **kwargs):
    return dataset_reader(batch, _folder(), 'trainB', **kwargs)


def dataset_testA(batch, **kwargs):
    return dataset_reader(batch, _folder(), 'testA', **kwargs)


def dataset_testB(batch, **kwargs):
    return dataset_reader(batch, _folder(), 'testB', **kwargs)


def _test_dataset():
    import sflow.py as py

    data = dataset_train(16)
    for d in tf.feeds(data):
        py.imshow([d.A, d.B])
        if not py.plot_pause():
            break



if __name__ == '__main__':
    _test_dataset()

