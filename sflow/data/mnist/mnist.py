# -*- coding: utf-8 -*-
"""
data from https://s3.amazonaws.com/img-datasets/mnist.npz
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/keras/python/keras/datasets/mnist.py
"""
import sflow.tf as tf
import sflow.py as py
import numpy as np


def _asset_folder():
    return tf.assets_folder('mnist/')


def dataset_train(batch, folder=None):

    (img, label), _ = _load_data(folder)
    with tf.name_scope('dataset_mnist'):
        # train.shape == (60000, 28, 28)
        img = tf.convert_to_tensor(img)
        label = tf.convert_to_tensor(label)
        ind = tf.random_uniform((batch,), minval=0, maxval=img.dims[0], dtype=tf.int32)

        x = tf.gather(img, ind)
        y = tf.gather(label, ind)

        q = tf.feed.queue_producer([x, y], capacity=100, shapes=[(batch, 28, 28), (batch,)],
                                   threads=1, shuffle=True)
        x, y = q.dequeue()
        x = x.expand_dims(3).to_float() / 255.
        y = y.to_int32()

    return tf.dic(image=x, label=y)


def dataset_test(batch, shuffle=False, folder=None):
    _, (img, label) = _load_data(folder)
    with tf.name_scope('dataset_mnist_test'):
        img = tf.convert_to_tensor(img)
        label = tf.convert_to_tensor(label)
        ind = tf.random_uniform((batch,), minval=0, maxval=img.dims[0], dtype=tf.int32)

        x = tf.gather(img, ind)
        y = tf.gather(label, ind)

        q = tf.feed.queue_producer([x, y], capacity=100, shapes=[(batch, 28, 28), (batch,)],
                                   threads=1, shuffle=shuffle)
        x, y = q.dequeue()
        x = x.expand_dims(3).to_float() / 255.
        y = y.to_int32()

    return tf.dic(image=x, label=y)


def _load_data(folder=None):
    import os
    folder = folder or _asset_folder()
    f = os.path.join(folder, 'mnist.npz')

    url = 'http://s3.amazonaws.com/img-datasets/mnist.npz'
    py.download_if_not(url, f)

    with np.load(f) as f:
        x_train = f['x_train']
        y_train = f['y_train']
        x_test = f['x_test']
        y_test = f['y_test']

    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':

    data = dataset_train(16)

    for d in tf.feeds(data):
        py.imshow(d.image)
        print(d.label)
        if not py.plot_pause():
            break
