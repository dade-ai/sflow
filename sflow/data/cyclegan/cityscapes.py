# -*- coding: utf-8 -*-
from .datasets import dataset_reader_AB
import sflow.tf as tf


def _folder():
    return 'cityscapes'


def dataset_train(batch, **kwargs):
    a = dataset_trainA(batch, **kwargs)
    b = dataset_trainB(batch, **kwargs)
    return tf.dic(A=a, B=b, batch=batch)


def dataset_test(batch, **kwargs):
    a = dataset_testA(batch, **kwargs)
    b = dataset_testB(batch, **kwargs)
    return tf.dic(A=a, B=b, batch=batch)


def dataset_valid(batch, **kwargs):
    a = dataset_valA(batch, **kwargs)
    b = dataset_valB(batch, **kwargs)
    return tf.dic(A=a, B=b, batch=batch)


def dataset_pair_train(batch, **kwargs):
    data = dataset_reader_AB(batch, _folder(), 'train', **kwargs).split(2, axis=2)

    return tf.dic(A=data[0], B=data[1], batch=batch)


def dataset_pair_test(batch, **kwargs):
    data = dataset_reader_AB(batch, _folder(), 'test', **kwargs).split(2, axis=2)

    return tf.dic(A=data[0], B=data[1], batch=batch)


def dataset_pair_val(batch, **kwargs):
    data = dataset_reader_AB(batch, _folder(), 'val', **kwargs).split(2, axis=2)

    return tf.dic(A=data[0], B=data[1], batch=batch)


def dataset_trainA(batch, **kwargs):
    data = dataset_reader_AB(batch, _folder(), 'train', **kwargs)
    return data.split(2, axis=2)[0]


def dataset_trainB(batch, **kwargs):
    data = dataset_reader_AB(batch, _folder(), 'train', **kwargs)
    return data.split(2, axis=2)[1]


def dataset_testA(batch, **kwargs):
    data = dataset_reader_AB(batch, _folder(), 'test', **kwargs)
    return data.split(2, axis=2)[0]


def dataset_testB(batch, **kwargs):
    data = dataset_reader_AB(batch, _folder(), 'test', **kwargs)
    return data.split(2, axis=2)[1]


def dataset_valA(batch, **kwargs):
    data = dataset_reader_AB(batch, _folder(), 'val', **kwargs)
    return data.split(2, axis=2)[0]


def dataset_valB(batch, **kwargs):
    data = dataset_reader_AB(batch, _folder(), 'val', **kwargs)
    return data.split(2, axis=2)[1]


def _test_dataset():
    import sflow.py as py

    data = dataset_valB(16)
    for d in tf.feeds(data):
        py.imshow(d)
        if not py.plot_pause():
            break


if __name__ == '__main__':
    _test_dataset()

