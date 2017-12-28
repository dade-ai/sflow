# -*- coding: utf-8 -*-
import sflow.tf as tf
import sflow.py as py


def default_folder():
    return tf.assets_folder('cyclegan/')


def dataset_reader(batch, dataset, partition, shape=(256, 256, 3),
                   capacity=100, preprocess=None, threads=4, shuffle=True, folder=None):
    import os

    folder = folder or default_folder()

    # pattern = '/data/vangogh2photo/trainA/*.jpg'
    pattern = os.path.join(folder, dataset, partition, '*.jpg')

    if not py.anyfile(pattern):
        if dataset in ['apple2orange', 'summer2winter_yosemite', 'horse2zebra',
                       'monet2photo', 'cezanne2photo', 'ukiyoe2photo', 'vangogh2photo',
                       'maps', 'cityscapes', 'facades', 'iphone2dslr_flower', 'ae_photos']:
            # todo check url reachable
            url = 'http://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/{}.zip'.format(dataset)
            zip = py.download(url, folder)
            py.unzip(zip)
            # remove file?
        else:
            url = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/{}.tar.gz'.format(dataset)
            tar = py.download(url, folder)
            py.untar(tar)
            # remove?

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
