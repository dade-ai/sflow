# -*- coding: utf-8 -*-
import os
import sflow.tf as tf


def _asset_folder():
    return tf.assets_folder('/imagenet/small/')


def dataset_train32(batch, **kwargs):
    shape = (32, 32, 3)
    img = _reader(batch, 'train_32x32', shape=shape, **kwargs)
    return img


def dataset_train64(batch, **kwargs):
    shape = (64, 64, 3)
    img = _reader(batch, 'train_64x64', shape=shape, **kwargs)
    return img


def dataset_val32(batch, shuffle=False, **kwargs):
    shape = (32, 32, 3)
    img = _reader(batch, 'valid_32x32', shape=shape, shuffle=shuffle, **kwargs)
    return img


def dataset_val64(batch, shuffle=False, **kwargs):
    shape = (64, 64, 3)
    img = _reader(batch, 'valid_64x64', shape=shape, shuffle=shuffle, **kwargs)
    return img


def _reader(batch, dataset, shape=None,
            capacity=100, preprocess=None, threads=4, shuffle=True, folder=None):
    import os

    folder = folder or _asset_folder()

    pattern = os.path.join(folder, dataset, '*.png')
    q = tf.feed.read_matching_image(pattern, shape=shape,
                                    capacity=capacity, preprocess=preprocess,
                                    threads=threads, shuffle=shuffle,
                                    channels=3)

    return q.dequeue_many(batch).to_float() / 255.

