# -*- coding: utf-8 -*-
"""
celeba dataset without label
"""
from __future__ import print_function
import os
import sflow.tf as tf


def _asset_folder():
    return tf.assets_folder('/face/celeba/')  #Img/img_align_celeba/'


def dataset(batch, size=None, crop=None, folder=None, capacity=512,
            threads=8, shuffle=None, partition='train', fliplr=False):

    folder = folder or _asset_folder()
    shuffle = shuffle or (partition == 'train')

    attrfile = 'list_attr_{0}.txt'.format(partition)
    trainfile = os.path.join(folder, 'Anno', attrfile)
    imgfolder = os.path.join(folder, 'Img/img_align_celeba/')
    imgfolder = tf.constant(imgfolder)

    with tf.name_scope('celeba'):
        line = tf.feed.read_line(trainfile)
        line = line.print('line', first_n=10)
        fields = tf.decode_csv(line, [['']] + [[-1]]*40, field_delim=' ')
        fname = fields[0]
        fname = imgfolder + fname
        img = tf.feed.read_image(fname, channels=3)

        # capacity = 512
        shapes = [(218, 178, 3)]
        q = tf.feed.queue_producer([img], capacity, shapes=shapes,
                                   threads=threads, shuffle=shuffle)
        img = q.dequeue_many(batch)
        # face only exclude attributes
        img = img.to_float()/255.
        img = tf.img.crop_center(crop or (178, 178), img)
        img = tf.image.resize_images(img, size or (256, 256))
        if fliplr:
            img = tf.img.rand_fliplr(img, p=0.5)

    return img


def dataset_train(batch, size=(256, 256), folder=None,
                  threads=8, shuffle=None, fliplr=False):
    """
    celeba dataset face only
    :param batch:
    :param size:
    :param folder:
    :param threads:
    :param shuffle:
    :param fliplr:
    :return:
    """
    return dataset(batch, size=size, folder=folder,
                   threads=threads, shuffle=shuffle,
                   partition='train', fliplr=fliplr)


def dataset_test(batch, folder=None, size=(256, 256),
                 threads=8, shuffle=None, fliplr=False):
    return dataset(batch, folder=folder, size=size,
                   threads=threads, shuffle=shuffle,
                   partition='test', fliplr=fliplr)


def dataset_valid(batch, folder=None, size=(256, 256),
                  threads=8, shuffle=None, fliplr=False):
    return dataset(batch, folder=folder, size=size,
                   threads=threads, shuffle=shuffle,
                   partition='valid', fliplr=fliplr)


def _test():
    import sflow.py as py

    data = dataset(16)
    with tf.feeding() as (sess, coord):
        while not coord.should_stop():
            d = sess.run(data)
            print(d.shape)
            py.plt.imshow(d)
            if not py.plt.plot_pause():
                break


if __name__ == '__main__':
    _test()
    pass

