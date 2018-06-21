# -*- coding: utf-8 -*-
from functools import wraps
import numpy as np
# sample images with BHWC float32 for test purpose tensor

# skimage.data.astronaut()	Colour image of the astronaut Eileen Collins.
# skimage.data.binary_blobs([length, ...])	Generate synthetic binary image with several rounded blob-like objects.
# skimage.data.camera()	Gray-level “camera” image.
# skimage.data.checkerboard()	Checkerboard image.
# skimage.data.chelsea()	Chelsea the cat.
# skimage.data.clock()	Motion blurred clock.
# skimage.data.coffee()	Coffee cup.
# skimage.data.coins()	Greek coins from Pompeii.
# skimage.data.horse()	Black and white silhouette of a horse.
# skimage.data.hubble_deep_field()	Hubble eXtreme Deep Field.
# skimage.data.img_as_bool(image[, force_copy])	Convert an image to boolean format.
# skimage.data.immunohistochemistry()	Immunohistochemical (IHC) staining with hematoxylin counterstaining.
# skimage.data.imread(fname[, as_grey, ...])	Load an image from file.
# skimage.data.load(f[, as_grey])	Load an image file located in the data directory.
# skimage.data.logo()	Scikit-image logo, a RGBA image.
# skimage.data.moon()	Surface of the moon.
# skimage.data.page()	Scanned page.
# skimage.data.rocket()	Launch photo of DSCOVR on Falcon 9 by SpaceX.
# skimage.data.stereo_motorcycle()	Rectified stereo image pair with ground-truth disparities.
# skimage.data.text()	Gray-level “text” image used for corner detection.


def _common_convert_deco(fn):
    import sflow.core as tf

    @wraps(fn)
    def _raise_on_exec():
        raise ValueError('check the version of skimage.data module')

    try:
        from skimage import data
        f = data.__dict__[fn.__name__]
    except KeyError:
        return _raise_on_exec

    @wraps(f)
    def _wraped(expand=False, size=None, dtype='float32', tensor=False):
        from skimage import io, transform

        img = f()
        if size is not None:
            sz = list(img.shape)
            sz[:len(size)] = size
            img = transform.resize(img, sz, preserve_range=True)
        if dtype == 'int8':
            pass
        elif dtype.startswith('float'):
            img = img.astype(dtype) / 255.
        if expand:
            img = np.expand_dims(img, 0)
            if img.ndim == 3:
                img = np.expand_dims(img, -1)
        if tensor:
            img = tf.convert_to_tensor(img)
        return img

    return _wraped


@_common_convert_deco
def astronaut():
    pass


@_common_convert_deco
def binary_blobs():
    pass


@_common_convert_deco
def camera():
    pass


@_common_convert_deco
def checkerboard():
    pass


@_common_convert_deco
def chelsea():
    pass


@_common_convert_deco
def clock():
    pass


@_common_convert_deco
def coffee(**kwargs):
    """
    :param bool expand: expand_axis(0) or not
    :param (int, int) size: resize or None
    :param str dtype:
    :param bool tensor:
    :return:
    :rtype (numpy|tensor)
    """
    pass


@_common_convert_deco
def coins():
    pass


@_common_convert_deco
def horse():
    pass


@_common_convert_deco
def hubble_deep_field():
    pass


@_common_convert_deco
def immunohistochemistry():
    pass


@_common_convert_deco
def imread():
    pass


@_common_convert_deco
def load():
    pass


@_common_convert_deco
def logo():
    pass


@_common_convert_deco
def moon():
    pass


@_common_convert_deco
def page():
    pass


@_common_convert_deco
def rocket():
    pass


@_common_convert_deco
def stereo_motorcycle():
    pass


@_common_convert_deco
def text():
    pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    c = binary_blobs()

    plt.imshow(c)
    plt.show()

