# -*- coding: utf-8 -*-
# http://www.kyprianidis.com/p/cag2012/winnemoeller-cag2012.pdf
# from __future__ import absolute_import
from .ssim import (gaussian_blur)

import sflow.core as tf
import snipy.common as py

# todo : FDoG, ETF(edge tangent flow)
# (no need for illustration but..., check it's possible with tf)


def DoG(x, sigma, k=1.6, window=3):
    # difference-of-Gaussians
    # D(x; sigma, k) = G(x; sigma) - G(x; k*sigma)
    # k = 1.6 by Marr and Hildreth

    g1 = gaussian_blur(x, window, sigma, padding='SAME')
    g2 = gaussian_blur(x, window, k*sigma, padding='SAME')

    return g1 - g2


def XDoG(x, p, pi=None, epsilon=None, sigma=1., k=1.6, window=3, name=None):
    """
    see : http://www.kyprianidis.com/p/cag2012/winnemoeller-cag2012.pdf

    example style:

        woodcut :  extreme edge emphasis settings to produce
        shape abstraction, and long, coherent carving-cuts (ϕ ≫ 0.01, σc ≈ 5 and p ≈ 100)

    Appendix A lists complete settings for many of our results,
    demonstrating the range over which we have found it useful
    to vary the XDoG parameters. We have found that choosing
    ε close to the midtone greyvalue of the image and p near 20,
    tends to lead to interesting stylizations;  though some specialized
    styles require much larger p values. The soft thresholding
    steepness parameter ϕ varies more widely. Because it controls
    the slope of the falloff, when ϕ is close to zero it is very sensitive
    to small changes, while the parameter becomes much less
    sensitive to small changes as it increases.

    :param x: image [bhwc]
    :param p: XDoG parameter p > 0.
    :param pi: threshold_ramp parameter, slope
    :param epsilon: threshold_ramp parameter. (0, 1), thresholding value
    :param sigma: DoG parameter p > 0.
    :param k: DoG parameter. k * sigma
    :param window: window size for gaussian filter
    :param name:
    :return:
    """

    # An eXtended difference-of-Gaussians
    # Reparameterization of the XDoG equation(7)
    # S(x; sigma, k, p) = G(x; sigma) + p*D(x; sigma, k)
    #                   = (1+p) * G(x; sigma) - p * G(x; k*sigma)

    k = tf.convert_to_tensor(k, dtype=tf.float32)
    g1 = gaussian_blur(x, window, sigma, padding='SAME')
    g2 = gaussian_blur(x, window, k*sigma, padding='SAME')

    out = (1. + p)*g1 - p*g2
    if epsilon is None and sigma is None:
        return out
    elif epsilon is not None and sigma is not None:
        return threshold_ramp(out, pi, epsilon, name=name or 'XDoG')
    else:
        raise ValueError('XDoG need both(for thresholding) or neither of (epsilon, pi), but epsilon {}, '
                         'pi {}'.format(epsilon, pi))


def threshold_ramp(x, pi, epsilon, name=None):
    # thresholding function Tε with a continuous ramp:

    return tf.where(x >= epsilon, tf.ones_like(x),
                    tf.tanh(pi * (x - epsilon)) + 1., name=name)


def _test_xdog_p_sigmas():
    from sflow.sample import astronaut

    img = astronaut(expand=True)
    x = tf.convert_to_tensor(img)

    n = 5
    sigmas = tf.linspace(1.0, 10.0, n)
    p_s = tf.linspace(0.1, 4.0, n)
    outputs = []
    for i in range(n):
        for j in range(n):
            out = XDoG(x, p=p_s[i], sigma=sigmas[j], window=11)
            out = tf.image.rgb_to_grayscale(out)
            out = tf.where(out > 0.6, tf.ones_like(out), tf.zeros_like(out))
            outputs.append(out)

    outputs = tf.concat(0, outputs)

    sess = tf.default_session()
    out = sess.run(outputs)

    py.plt.imshow(out, cmap='gray')
    py.plt.plot_pause()


def _test_xdog_pi_epsilon():
    from sflow.sample import astronaut

    img = astronaut(expand=True)
    # img = thumbnail_room(expand=True)
    x = tf.convert_to_tensor(img)

    n = 10
    p = 20.
    sigma = 4
    eps_s = tf.linspace(0.2, 0.9, n)
    pi_s = tf.linspace(0.4, 30.0, n)
    outputs = []
    x = tf.image.rgb_to_grayscale(x)

    for i in range(n):
        for j in range(n):
            out = XDoG(x, p=p, sigma=sigma, window=11, epsilon=eps_s[i], pi=pi_s[j])
            outputs.append(out)

    out = tf.concat(0, outputs)

    sess = tf.default_session()
    o = sess.run(out)

    py.plt.imshow(o, cmap='gray')
    py.plt.plot_pause()


if __name__ == '__main__':
    # _test_xdog_p_sigmas()
    _test_xdog_pi_epsilon()
