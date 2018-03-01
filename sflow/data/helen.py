# -*- coding: utf-8 -*-
# data from
# http://pages.cs.wisc.edu/~lizhang/projects/face-parsing/
import sflow.tf as tf


def datasetfolder():
    return tf.assets_folder('/face/helen/SmithCVPR2013_dataset_resized/')


def _trainfile_old(folder=None):
    import os
    return os.path.join(folder or datasetfolder(), 'trainnames.txt')


def _trainfile(folder=None):
    import os
    return os.path.join(folder or datasetfolder(), 'exemplars.txt')


def _testfile(folder=None):
    import os
    return os.path.join(folder or datasetfolder(), 'testing.txt')


def read_label(fid, folder=None):
    import sflow.tf as tf
    import os
    fname = tf.constant(os.path.join(folder or datasetfolder(), 'labels/'))
    post = ['_lbl{:02d}.png'.format(i) for i in range(11)]

    f = fname + fid + '/' + fid + post

    imgs = []
    for i, p in enumerate(post):
        data = tf.read_file(f[i])
        img = tf.image.decode_image(data)
        img.set_shape((None, None, 1))
        imgs.append(img)

    # return labels
    labels = tf.concat(2, imgs)
    labels = tf.Print(labels, [fid] + [tf.shape(l) for l in imgs], 'labels', first_n=10)

    return labels


def normalize_label(label, name=None):
    import sflow.tf as tf
    dims = label.dims
    s = label.sum(axis=-1, keepdims=True)
    dims[-1] -= 1
    bg = tf.concat(-1, [s.equal(0.), tf.zeros(dims, dtype=tf.bool)])
    label = tf.where(bg, tf.ones(label.dims), label)

    # label[:,:,:,0] = tf.where(s.equal(0.), tf.ones(dims), label[:,:,:,0])
    s = label.sum(axis=-1, keepdims=True)
    label = label / s

    return tf.identity(label, name=name or 'label')


def faceklass_desc():
    return ['background', 'face_skin', 'left_eyebrow', 'right_eyebrow', 'left_eye',
            'right_eye', 'nose', 'upper_lip', 'inner_mouth', 'lower_lip', 'hair']


def feed_img_label(fid, size=(256, 256), shuffle=False, rotate=True, threads=6, folder=None):
    import os
    import sflow.tf as tf

    folder = tf.convert_to_tensor(os.path.join(folder or datasetfolder(), 'images/'))

    fid = tf.feed.queue_producer(fid, 1000, shuffle=shuffle, min_after_dequeue=20).dequeue()
    imgfile = folder + fid + '.jpg'

    label = read_label(fid)
    img = tf.feed.read_image(imgfile, channels=3)

    # preprocess
    szbig = list(map(int, (256*1.5, 256*1.5)))

    img = tf.img.pad_if_need(img, size=szbig)
    label = tf.img.pad_if_need(label, size=szbig)

    img, label = tf.img.rand_crop3d(szbig, img, label)
    # img.set_shape([szbig[0], szbig[1], 3])

    img = tf.expand_dims(img, axis=0)
    label = tf.expand_dims(label, axis=0)

    if rotate:
        img, label = tf.img.rand_rotate(img, label, angles=(-tf.pi*.25, tf.pi*.25))
    img, label = tf.img.crop_center(size, img, label)

    img = img.squeeze(0)
    label = label.squeeze(0)

    # end of preprocess
    q = tf.feed.queue_producer([img, label], 200, shuffle=shuffle, min_after_dequeue=10, threads=threads)

    return q


def trainset_old(batch, size=(256, 256), threads=8, sizedown=None, removebg=False):
    import sflow.tf as tf

    # qtrain : from train
    with tf.name_scope('helen_trainset'):
        fid = tf.feed.read_line(_trainfile(), shuffle=True)
        qtrain = feed_img_label(fid, size, shuffle=True, threads=threads)

        image, label = qtrain.dequeue_many(batch)
        image = image.to_float()/255.
        image = tf.identity(image, name='image')
        label = label.to_float()/255.

        if sizedown is not None:
            image = image.sizedown(sizedown)
            label = label.sizedown(sizedown)
        label = normalize_label(label)

        if removebg is True:
            # remove background image
            bg = 1. - label[:, :, :, :1]
            image = image * bg

    return tf.dic(image=image, label=label, batch=batch)


def trainset(batch, size=(256, 256), threads=8, sizedown=None, removebg=False):
    import sflow.tf as tf

    # qtrain : from train
    with tf.name_scope('helen_trainset'):
        # fid = tf.feed.read_line(_trainfile(), shuffle=True)
        line = tf.feed.read_line(_trainfile(), shuffle=True)
        _, fid = tf.decode_csv(line, [[0], ['']], field_delim=',')
        trimed = tf.string_split([fid], delimiter=' ')  # remove space
        # now trimed.values[0] has fid
        fid = trimed.values[0]  # ex) '100032540_1'

        qtrain = feed_img_label(fid, size, shuffle=True, threads=threads)

        image, label = qtrain.dequeue_many(batch)
        image = image.to_float()/255.
        image = tf.identity(image, name='image')
        label = label.to_float()/255.

        if sizedown is not None:
            image = image.sizedown(sizedown)
            label = label.sizedown(sizedown)
        label = normalize_label(label)

        if removebg is True:
            # remove background image
            bg = 1. - label[:, :, :, :1]
            image = image * bg

    return tf.dic(image=image, label=label)


def testset(batch, size=(256, 256), threads=8, sizedown=None, removebg=False):
    import sflow.tf as tf

    with tf.name_scope('helen_testset'):
        # qtest : from testing.txt format , seperated
        line = tf.feed.read_line(_testfile())
        _, fid = tf.decode_csv(line, [[0], ['']], field_delim=',')
        trimed = tf.string_split([fid], delimiter=' ')  # remove space
        # now trimed.values[0] has fid
        fid = trimed.values[0]  # ex) '100032540_1'
        qtest = feed_img_label(fid, size, shuffle=False, rotate=False, threads=threads)

        data = qtest.dequeue_many(batch)
        image = data[0].to_float()/255.
        label = data[1].to_float()/255.

        if sizedown is not None:
            image = image.sizedown(sizedown)
            label = label.sizedown(sizedown)
        label = normalize_label(label)

        if removebg is True:
            # remove background image
            bg = 1. - label[:, :, :, :1]
            image = image*bg

    return tf.dic(image=image, label=label)


def label_to_rgb(label):
    """
    utility for viewing label in a glance
    :param label: [b x h x w x 11]
    :return: [b x h x w x 3]
    """
    pass


if __name__ == '__main__':
    import sflow.tf as tf
    d = trainset(16)
    import matplotlib.pyplot as plt

    with tf.feeding() as (sess, coord):
        while not coord.should_stop():
            out = sess.run(d)

            print(out.image.shape)
            print(out.label.shape)

            plt.imshow(out.image[0])
            plt.show()

