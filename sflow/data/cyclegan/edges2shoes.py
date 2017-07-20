# -*- coding: utf-8 -*-
from .datasets import dataset_reader
import sflow.tf as tf


def _folder():
    return 'edges2shoes'


def dataset_train(batch, **kwargs):
    a = dataset_pair(batch, 'train', shuffle=True, **kwargs)
    b = dataset_pair(batch, 'train', shuffle=True, **kwargs)

    return tf.dic(A=a.A, B=b.B, batch=batch)


def dataset_pair_train(batch, **kwargs):
    return dataset_pair(batch, 'train', shuffle=True, **kwargs)


def dataset_val(batch, **kwargs):
    """
    read validation pair dataset
    :param batch:
    :param kwargs:
    :return:
    """
    return dataset_pair(batch, 'val', shuffle=False, **kwargs)


def dataset_pair(batch, partition, **kwargs):
    shape = (256, 512, 3)
    data = dataset_reader(batch, _folder(), partition, shape=shape, **kwargs)
    a, b = tf.split(data, 2, axis=2)
    return tf.dic(A=a, B=b, batch=batch)


def _test_dataset():
    import sflow.py as py

    data = dataset_train(16)
    for d in tf.feeds(data):
        py.imshow([d.A, d.B])
        if not py.plot_pause():
            break

if __name__ == '__main__':
    _test_dataset()

