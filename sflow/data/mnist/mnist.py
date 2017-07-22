# -*- coding: utf-8 -*-
"""
data from https://s3.amazonaws.com/img-datasets/mnist.npz
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/keras/python/keras/datasets/mnist.py
"""
import sflow.tf as tf
import sflow.py as py
import numpy as np
import os


def _asset_folder():
    return tf.assets_folder('mnist/')


def dataset_train(batch, size=(28, 28), folder=None):

    (img, label), _ = _load_data(size, folder)
    with tf.name_scope('dataset_mnist'):
        # train.shape == (60000, 28, 28)

        img = tf.constant(img)
        label = tf.constant(label)
        ind = tf.random_uniform((batch,), minval=0, maxval=img.dims[0], dtype=tf.int32)

        x = tf.gather(img, ind)
        y = tf.gather(label, ind)

        if x.ndim == 3:
            x = x.expand_dims(3)
        if x.dtype != tf.float32:
            x = x.to_float() / 255.

        y = y.to_int32()

    return tf.dic(image=x, label=y)


def dataset_test(batch, shuffle=True, size=(28, 28), folder=None):

    if not shuffle:
        raise NotImplementedError

    _, (img, label) = _load_data(size, folder)
    with tf.name_scope('dataset_mnist_test'):
        img = tf.constant(img)
        label = tf.constant(label)
        ind = tf.random_uniform((batch,), minval=0, maxval=img.dims[0], dtype=tf.int32)

        x = tf.gather(img, ind)
        y = tf.gather(label, ind)

        if x.ndim == 3:
            x = x.expand_dims(3)
        if x.dtype != tf.float32:
            x = x.to_float() / 255.

        y = y.to_int32()

    return tf.dic(image=x, label=y)


def _resize_mnist_image(d, size):
    from skimage import transform

    d = d.transpose((1, 2, 0))
    d = transform.resize(d, size)
    d = d.astype('float32')
    d = d.transpose((2, 0, 1))

    return d


def _load_numpy(f):
    with np.load(f) as f:
        x_train = f['x_train']
        y_train = f['y_train']
        x_test = f['x_test']
        y_test = f['y_test']
    return (x_train, y_train), (x_test, y_test)


def _load_resized_local(folder, size):
    if size == (28, 28):
        return _load_numpy(os.path.join(folder, 'mnist.npz'))

    f = os.path.join(folder, 'mnist.{}x{}.npz'.format(*size))

    if not os.path.exists(f):
        (x_train, y_train), (x_test, y_test) = _load_numpy(os.path.join(folder, 'mnist.npz'))
        x_train = _resize_mnist_image(x_train, size)
        x_test = _resize_mnist_image(x_test, size)
        np.savez(f, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        return (x_train, y_train), (x_test, y_test)
    else:
        return _load_numpy(f)


def _load_data(size=(28, 28), folder=None):
    folder = folder or _asset_folder()

    f = os.path.join(folder, 'mnist.npz')
    url = 'http://s3.amazonaws.com/img-datasets/mnist.npz'
    py.download_if_not(url, f)

    return _load_resized_local(folder, size)


if __name__ == '__main__':
    data = dataset_train(16, (7, 7))

    for d in tf.feeds(data):
        py.imshow(d.image)
        print(d.label)
        if not py.plot_pause():
            break
