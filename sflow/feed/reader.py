# -*- coding: utf-8 -*-
import tensorflow as tf


def _helper_img_decode(decoder, fname, shape=(None, None, None),
                       verbose=False, shuffle=False, **kwargs):

    if hasattr(fname, 'queue_ref'):
        imreader = tf.WholeFileReader()
        key, content = imreader.read(fname)
    elif isinstance(fname, (tuple, list)) or fname.ndim >= 1:
        fname = tf.train.string_input_producer(fname, shuffle=shuffle)
        imreader = tf.WholeFileReader()
        key, content = imreader.read(fname)
    elif isinstance(fname, str):
        content = tf.read_file(fname)
    elif isinstance(fname, tf.Tensor):
        if fname.ndim == 0:
            content = tf.read_file(fname)
        else:
            fname = tf.train.string_input_producer(fname, shuffle=shuffle)
            imreader = tf.WholeFileReader()
            key, content = imreader.read(fname)
    else:
        raise ValueError('what input for read? {0}'.format(fname))

    img = decoder(content, **kwargs)
    channels = kwargs.pop('channels', None)
    img.set_shape([shape[0], shape[1], shape[2] or channels])

    if verbose:
        return tf.Print(img, [fname, tf.shape(img)], message='reading image of ')

    return img


def read_image(fname, shape=(None, None, None), channels=None,
               verbose=False, shuffle=False, **kwargs):
    """
    return uint8 image
    :param fname:
    :param shape:
    :param channels:
    :param verbose:
    :param shuffle:
    :param kwargs:
    :return:
    """
    return _helper_img_decode(tf.image.decode_image, fname, shape=shape,
                              channels=channels, verbose=verbose,
                              shuffle=shuffle, **kwargs)


def read_jpeg(fname, shape=(None, None, None), channels=None,
              verbose=False, shuffle=False, **kwargs):
    return _helper_img_decode(tf.image.decode_jpeg, fname, shape=shape,
                              channels=channels, verbose=verbose,
                              shuffle=shuffle, **kwargs)


def read_png(fname, shape=(None, None, None), channels=None,
             verbose=False, shuffle=False, **kwargs):
    return _helper_img_decode(tf.image.decode_png, fname, shape=shape,
                              channels=channels, verbose=verbose,
                              shuffle=shuffle, **kwargs)


def read_gif(fname, shape=(None, None, None),
             verbose=False, shuffle=False, **kwargs):
    return _helper_img_decode(tf.image.decode_gif, fname, shape=shape,
                              verbose=verbose,
                              shuffle=shuffle, **kwargs)


def read_line(fname, up_to=None, shuffle=False):
    """
    todo example
    :param fname:
    :param up_to:
    :param shuffle:
    :return:
    """
    if isinstance(fname, (str, tf.Tensor)):
        fname = tf.train.string_input_producer([fname], shuffle=shuffle)
        # fname = tf.convert_to_tensor(fname, dtype=tf.string)
    reader = tf.TextLineReader()
    if up_to is None:
        _, line = reader.read(fname)
    else:
        _, line = reader.read_up_to(fname, num_records=up_to)

    return line

