# -*- coding: utf-8 -*-
import sflow.tf as tf


def _folder():
    return tf.assets_folder('/svhn/')


def dataset_train(batch, shape=None, folder=None):
    import os
    from scipy.io import loadmat

    # load .mat file
    folder = folder or _folder()
    fpath = os.path.join(folder, 'train_32x32.mat')
    data = loadmat(fpath)
    x = data['X']  #
    y = data['y']  # 1~10 ?

    x = x.transpose((3, 0, 1, 2))
    y[y == 10] = 0
    n = x.shape[0]  # total data

    # convert to tensors
    x = tf.convert_to_tensor(x)  # uint8 (73257, 32, 32, 3)
    y = tf.convert_to_tensor(y)  # uint8 (73257, 1)

    # random select
    i = tf.random_uniform((batch,), minval=0, maxval=n, dtype=tf.int32)
    img = tf.gather(x, i).to_float()/255.
    label = tf.gather(y, i).to_int32()

    if shape is not None:
        img = tf.image.resize_images(img, shape)

    return tf.dic(image=img, label=label)


# def dataset_test():
#     pass


# if __name__ == '__main__':
#     import sflow.py as py
#     img, label = dataset_train(4)
#
#     sess = tf.default_session()
#
#     for i in range(50):
#         res = sess.run([img, label])
#         py.plt.imshow(res[0])
#         print(res[1])
#         if not py.plt.plot_pause():
#             break

