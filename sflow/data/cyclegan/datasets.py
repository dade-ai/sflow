# -*- coding: utf-8 -*-
import sflow.tf as tf


def default_folder():
    return '/data/cyclegan/'


def dataset_reader(batch, dataset, partition, shape=(256, 256, 3),
                   capacity=100, preprocess=None, threads=4, shuffle=True, folder=None):
    import os

    folder = folder or default_folder()

    # pattern = '/data/vangogh2photo/trainA/*.jpg'
    pattern = os.path.join(folder, dataset, partition, '*.jpg')
    q = tf.feed.read_matching_image(pattern, shape=shape,
                                    capacity=capacity, preprocess=preprocess,
                                    threads=threads, shuffle=shuffle,
                                    channels=3)

    return q.dequeue_many(batch).to_float() / 255.


def dataset_reader_AB(batch, dataset, partition, shape=(256, 512, 3),
                      capacity=100, preprocess=None, threads=4, shuffle=True, folder=None):
    return dataset_reader(batch, dataset, partition, shape=shape,
                          capacity=100, preprocess=None, threads=4, shuffle=True,
                          folder=None)
