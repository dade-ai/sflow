# -*- coding: utf-8 -*-
# data feeder for attribute pair (positive, negative pair) for given attribute category

from __future__ import print_function
import os
import sflow.tf as tf


def datasetfolder():
    return tf.assets_folder('/face/celeba/')  #Img/img_align_celeba/'


def attributes_desc():
    """
    return list of column descriptions
    5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs
    Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows
    Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones
    Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin
    Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair
    Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young

    :return: list[str]
    """
    columns = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',
        'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
        'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
        'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
        'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young',
    ]

    return map(str.lower, columns)


# region preprocessing

def _attr_column_index(attr):
    attr = attr.lower()
    for i, a in enumerate(attributes_desc()):
        if a == attr:
            return i, attr
    raise ValueError('unknown attribute name')


def _pair_list_file(attr, folder=None, partition='train'):
    folder = folder or datasetfolder()
    pairfiles = ['list_attr_{0}.{1}.{2}.txt'.format(partition, attr, i) for i in [0, 1]]
    pairfiles = [os.path.join(folder, 'Anno', f) for f in pairfiles]
    return pairfiles


def _prepare_attr_pair_list(attr, folder=None, partition='train'):
    folder = folder or datasetfolder()
    attrfile = 'list_attr_{0}.txt'.format(partition)
    attrfile = os.path.join(folder, 'Anno', attrfile)
    # imgfolder = os.path.join(folder, 'Img/img_align_celeba/')
    icolumn, attr = _attr_column_index(attr)

    icolumn += 1  # consider filename index

    pairfiles = ['list_attr_{0}.{1}.{2}.txt'.format(partition, attr, i) for i in [0, 1]]
    pairfiles = [os.path.join(folder, 'Anno', f) for f in pairfiles]
    pairfiles = [open(f, 'w') for f in pairfiles]

    lines = [[], []]
    with open(attrfile, 'r') as f:
        for line in f:
            fname_attrs = line.split()
            if fname_attrs[icolumn] == '-1':
                # pairfiles[0].write(line)
                lines[0].append(line)
            else:
                # pairfiles[1].write(line)
                lines[1].append(line)

    # align attribute pair size
    print('lines i{0}'.format(map(len, lines)))
    n = min(map(len, lines))

    for f, flines in zip(pairfiles, lines):
        for i in range(n):
            f.write(flines[i])
        f.close()
        print('attribute written to [{0}]'.format(f.name))

    print('done')


def attr_dataset(batch, attr, value, size=None, crop=None,
                 threads=8, shuffle=None,
                 partition='train', folder=None, fliplr=False):
    """
    get data set given attribute data
    :param batch:
    :param attr:
    :param value:
    :param size:
    :param crop:
    :param threads:
    :param shuffle:
    :param partition:
    :param folder:
    :param fliplr:
    :return:
    """
    folder = folder or datasetfolder()
    shuffle = shuffle or (partition == 'train')
    attr = attr.lower()
    files = _pair_list_file(attr, folder=folder, partition=partition)
    if not (os.path.exists(files[0]) and os.path.exists(files[1])):
        _prepare_attr_pair_list(attr, folder=folder, partition=partition)

    if value is False:
        filelist = files[0]
    else:
        filelist = files[1]

    imgfolder = os.path.join(folder, 'Img/img_align_celeba/')
    imgfolder = tf.constant(imgfolder)

    with tf.name_scope(None, 'celeba.attr.{0}.{1}'.format(attr, value)):
        line = tf.feed.read_line(filelist)
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

        img = img.to_float()/255.
        crop = crop or (178, 178)
        if crop is not None:
            img = tf.img.crop_center(crop, img)
        if size is not None:
            img = tf.image.resize_images(img, size)
        if fliplr:
            img = tf.img.rand_fliplr(img, p=0.5)

    return tf.dic(image=img, label=attrs, batch=batch)


def attr_pair(batch, attr, size=None, crop=None, threads=8,
              shuffle=None, partition='train',
              folder=None, fliplr=False):
    size = size or (128, 128)

    data0 = attr_dataset(batch, attr, value=False, size=size, crop=crop, threads=threads,
                         shuffle=shuffle, partition=partition,
                         folder=folder, fliplr=fliplr)

    data1 = attr_dataset(batch, attr, value=True, size=size, crop=crop, threads=threads,
                         shuffle=shuffle, partition=partition,
                         folder=folder, fliplr=fliplr)

    return tf.dic(x0=data0.image, x1=data1.image)


# if __name__ == '__main__':
#     import sflow.gpu0 as tf
#     import matplotlib.pyplot as plt
#
#     trainset = attr_pair(4, attr='Eyeglasses')
#
#     with tf.feeding() as (sess, coord):
#         while not coord.should_stop():
#             img0, img1 = sess.run([trainset.image0, trainset.image1])
#
#             print(img0.shape)
#             print(img1.shape)
#             plt.subplot(1, 2, 1)
#             plt.imshow(img0[0])
#             plt.subplot(1, 2, 2)
#             plt.imshow(img1[0])
#             plt.show()
#
