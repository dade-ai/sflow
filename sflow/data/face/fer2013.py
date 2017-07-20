# -*- coding: utf-8 -*-
import sflow.tf as tf
import sflow.py as py


def datasetfolder():
    return tf.assets_folder('/face/fer2013')


def datasetfile():
    import os
    return os.path.join(datasetfolder(), 'fer2013.pkl')


def emotion_desc():
    return ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


def usages_dic():
    return {'PublicTest': 0, 'Training': 1, 'PrivateTest': 2}


def _convert_csv_to_pkl(folder=None):
    import os
    import csv
    import numpy as np

    # import snipy.ploting as ploting
    # import matplotlib.pyplot as plt
    folder = folder or datasetfolder()
    fpath = os.path.join(folder, 'fer2013.csv')
    # emotiondic = emotion_desc()
    usagesdic = usages_dic()
    images, emotions, usages = [], [], []
    with open(fpath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        reader.next()  # skip header ['emotion', 'pixels', 'Usage']
        for i, rows in py.iprogress(reader):
            emotion, pixels, usage = rows
            emotion = int(emotion)
            img = np.asarray([int(p) for p in pixels.split()], dtype=np.uint8)
            img = np.reshape(img, (48, 48, 1))
            iusage = usagesdic[usage]

            images.append(img)
            emotions.append(emotion)
            usages.append(iusage)

            # plt.imshow(img, cmap='Greys_r')
            # plt.title(emotions[emotion])
            # ploting.plot_pause()

            # if i > 10:
            #     break
    data = dict()
    data['faces'] = np.asarray(images, dtype=np.uint8)
    data['emotions'] = np.asarray(emotions, dtype=np.uint8)
    data['usages'] = np.asarray(usages, dtype=np.uint8)

    # no compression for mem
    return py.io.savefile(data, datasetfile(), compress=False)


def loaddata(mmap_mode='r', folder=None):
    """
    load FER2013 data returns faces, emotions
    :param mmap_mode: {None, ‘r+’, ‘r’, ‘w+’, ‘c’} see. joblib.load
    :return: faces, emotions
    """
    folder = folder or datasetfile()
    data = py.io.loadfile(folder, mmap_mode=mmap_mode)
    if data is None:
        data = _convert_csv_to_pkl()
    faces = data['faces']
    emotions = data['emotions']

    # (35887, 48, 48) uint8, (35887,) uint8
    return faces, emotions


@tf.feed.iterable
def gen_random_face(folder=None):
    faces, emotions = loaddata(folder)
    count = emotions.shape[0]

    while True:
        i = 0
        while i < count:
            yield faces[i]/255., emotions[i]
            i += 1


def dataset_emotion(batch, threads=8, shuffle=None, capacity=10, folder=None):
    """

    :param batch:
    :param threads:
    :param shuffle:
    :param capacity:
    :param folder:
    :return: dict(image, label)
    """

    with tf.name_scope('fer2013.emotion'):
        # face image 48x48x1
        img = tf.placeholder(tf.float32, shape=(48, 48, 1), name='image')
        # emotion label
        label = tf.placeholder(tf.int32, shape=(), name='emotion')
        placeholders = [img, label]
        q = tf.feed.gen_producer(placeholders, gen_random_face(folder),
                                 capacity=capacity, threads=threads, shuffle=shuffle)

        d = q.dequeue_many(batch)

    return tf.dic(image=d[0], label=d[1])


if __name__ == '__main__':
    # _convert_csv_to_pkl()

    data = dataset_emotion(12)
    with tf.feeding() as (sess, coord):
        while not coord.should_stop():
            out = sess.run(data)
            py.plt.imshow(out.image)
            py.plt.plot_pause()




