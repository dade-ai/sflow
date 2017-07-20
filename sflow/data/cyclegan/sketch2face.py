# -*- coding: utf-8 -*-

# celeba face preprocessed

from .datasets import dataset_reader
import sflow.tf as tf


def _asset_folder():
    return tf.assets_folder('/face/celeba')


def _folder():
    return 'process'


def dataset_train(batch, **kwargs):
    """
    sketch collection to face photo collection
    :param batch:
    :param kwargs:
    :return:
    """
    a = dataset_trainA(batch, **kwargs)
    b = dataset_trainB(batch, **kwargs)
    return tf.dic(A=a, B=b, batch=batch)


def process_256x256(x):
    """
    [n, 218,178,3] -> [n, 256, 256, 3]
    :param x:
    :return:
    """
    x = tf.img.crop_center((178, 178), x)
    x = tf.image.resize_images(x, (256, 256))
    return x


def dataset_trainA(batch, **kwargs):
    shape = (218, 178, 3)
    data = dataset_reader(batch, _folder(), 'sketch', folder=_asset_folder(),
                          shape=shape, **kwargs)
    return process_256x256(data)


def dataset_trainB(batch, **kwargs):
    shape = (218, 178, 3)
    data = dataset_reader(batch, _folder(), 'resized', folder=_asset_folder(),
                          shape=shape, **kwargs)
    return process_256x256(data)


def dataset_pair_train():
    raise NotImplementedError


def dataset_pair_test():
    raise NotImplementedError


def _test_dataset():
    import sflow.py as py

    data = dataset_train(16)
    for d in tf.feeds(data):
        py.imshow([d.A, d.B])
        if not py.plot_pause():
            break



if __name__ == '__main__':
    _test_dataset()


