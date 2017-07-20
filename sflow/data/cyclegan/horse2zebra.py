# -*- coding: utf-8 -*-
from .datasets import dataset_reader
import sflow.tf as tf


def _folder():
    return 'horse2zebra'


def dataset_train(batch, **kwargs):
    a = dataset_trainA(batch, **kwargs)
    b = dataset_trainB(batch, **kwargs)

    return tf.dic(A=a, B=b, batch=batch, size=(939, 1177))


def dataset_test(batch, **kwargs):
    a = dataset_trainA(batch, **kwargs)
    b = dataset_trainB(batch, **kwargs)
    return tf.dic(A=a, B=b, batch=batch)


def dataset_trainA(batch, shuffle=True, **kwargs):
    return dataset_reader(batch, _folder(), 'trainA', shuffle=shuffle, **kwargs)


def dataset_trainB(batch, shuffle=True, **kwargs):
    return dataset_reader(batch, _folder(), 'trainB', shuffle=shuffle, **kwargs)


def dataset_testA(batch, shuffle=False, **kwargs):
    return dataset_reader(batch, _folder(), 'testA', shuffle=shuffle, **kwargs)


def dataset_testB(batch, shuffle=False, **kwargs):
    return dataset_reader(batch, _folder(), 'testB', shuffle=shuffle, **kwargs)


def _test_dataset():
    import sflow.py as py

    data = dataset_train(16)
    for d in tf.feeds(data):
        py.imshow([d.A, d.B])
        if not py.plot_pause():
            break


if __name__ == '__main__':
    _test_dataset()

