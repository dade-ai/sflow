# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sflow.tf as tf


def datasetfolder():
    return tf.assets_folder('/face/celeba/')   #Img/img_align_celeba/'

# region preprocessing


def _partition_dataset(folder=None):
    # split 'Anno/list_eval_partition.txt' to
    # Anno/list_train.txt
    # Anno/list_valid.txt
    # Anno/list_test.txt
    folder = folder or datasetfolder()

    files = ['list_train.txt', 'list_valid.txt', 'list_test.txt']
    files = [os.path.join(folder, 'Eval', f) for f in files]
    files = [open(f, 'w') for f in files]

    f = os.path.join(folder, 'Eval', 'list_eval_partition.txt')
    with open(f, 'r') as f:
        for line in f:
            fname, i = line.split()
            # 0 train, 1 valid, 2 test
            files[int(i)].write(fname + '\n')

    for f in files:
        f.close()
    print('done')


def _load_partition_dict(folder=None):
    folder = folder or datasetfolder()

    f = os.path.join(folder, 'Eval', 'list_eval_partition.txt')
    partition_dict = dict()
    with open(f, 'r') as f:
        for line in f:
            fname, i = line.split()
            # 0 train, 1 valid, 2 test
            partition_dict[fname] = int(i)

    return partition_dict


def _partition_attribute(folder=None):
    # 5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs
    # Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows
    # Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones
    # Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin
    # Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair
    # Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young

    folder = folder or datasetfolder()
    partition_dict = _load_partition_dict(folder)
    files = ['list_attr_train.txt', 'list_attr_valid.txt', 'list_attr_test.txt']
    files = [os.path.join(folder, 'Anno', f) for f in files]
    files = [open(f, 'w') for f in files]

    f = os.path.join(folder, 'Anno', 'list_attr_celeba.txt')
    with open(f, 'r') as f:
        next(f)  # skip # of images
        next(f)  # skip header
        for line in f:
            fname = line.split()[0]
            i = partition_dict[fname]
            line = ' '.join(l for l in line.split() if l) + '\n'
            files[i].write(line)

    for f in files:
        f.close()
    print('done')


# endregion

# region util


def attribute_desc(folder=None):
    folder = folder or datasetfolder()
    f = os.path.join(folder, 'Anno', 'list_attr_celeba.txt')
    with open(f, 'r') as f:
        f.readline()
        desc = f.readline().split()
    return desc

# endregion

# region attribute tensor datasets


def attribute_dataset(batch, size=None, crop=None, threads=8, shuffle=None,
                      partition='train', folder=None, fliplr=False):

    folder = folder or datasetfolder()
    # assert size is None  # not implemented size

    attrfile = 'list_attr_{0}.txt'.format(partition)
    shuffle = shuffle or (partition == 'train')
    trainfile = os.path.join(folder, 'Anno', attrfile)
    imgfolder = os.path.join(folder, 'Img/img_align_celeba/')
    # imgfolder = os.path.join(folder, 'process/resized/')

    imgfolder = tf.constant(imgfolder)

    with tf.name_scope('celeba_attribute'):
        line = tf.feed.read_line(trainfile)

        line = line.print('line', first_n=10)

        fields = tf.decode_csv(line, [['']] + [[-1]]*40, field_delim=' ')
        fname = fields[0]
        fname = imgfolder + fname
        img = tf.feed.read_image(fname, channels=3)

        attrs = tf.stack(fields[1:])
        attrs = tf.equal(attrs, 1).to_float()

        capacity = 512
        shapes = [(218, 178, 3), (40,)]
        q = tf.feed.queue_producer([img, attrs], capacity, shapes=shapes,
                                   threads=threads, shuffle=shuffle)
        img, attrs = q.dequeue_many(batch)
        # img = tf.img.rand_fliplr(img, p=0.5)

        img = img.to_float()/255.
        if size is not None:
            img = tf.image.resize_images(img, size)
        if crop is not None:
            img = tf.img.crop_center(crop, img)
        if fliplr:
            img = tf.img.rand_fliplr(img, p=0.5)

    return tf.dic(image=img, label=attrs, batch=batch)


def attribute_trainset(batch, size=None, crop=None, threads=8):
    """
    celabA attribute training dataset
    :param batch:
    :param size: size to crop center
    :param crop:
    :param threads:
    :return:
    """
    return attribute_dataset(batch, size=size, crop=crop, threads=threads, partition='train')


def attribute_validset(batch, size=None, threads=8):
    return attribute_dataset(batch, size=size, threads=threads, partition='valid')


def attribute(batch, size=None, threads=8):
    """
    add comment size, fields
    :param batch:
    :param size:
    :param threads:
    :return:
    """

    return tf.dic(train=attribute_dataset(batch, size=size, threads=threads, partition='train'),
                  valid=attribute_dataset(batch, size=size, threads=threads, partition='valid'),)


# endregion

if __name__ == '__main__':
    # _partition_attribute()
    # d = attribute_desc()
    # print(d)
    # print(len(d))
    # from sflow import tf
    import sflow.gpu0 as tf
    import matplotlib.pyplot as plt

    trainset = attribute_trainset(6, size=(192, 168))

    with tf.feeding() as (sess, coord):
        while not coord.should_stop():
            img, label = sess.run([trainset.image, trainset.label])

            print(img.shape)
            print(label.shape)
            plt.imshow(img[0])
            plt.show()

