# -*- coding: utf-8 -*-
import sflow.gpu1 as tf


# @tf.scope
# def generator(z):
#     deconv = dict(kernel=4, stride=2, padding='SAME')
#
#     with tf.default_args(deconv=deconv):
#         net = z
#         net = net.dense(4*3*512).bn().relu()
#         net = net.reshape((-1, 4, 3, 512))  # (4, 3)
#         net = net.deconv(256).bn().relu()   # (8, 6)
#         net = net.deconv(128).bn().relu()   # (16, 12)
#         net = net.deconv(64).bn().relu()    # (32, 24)
#         net = net.deconv(32).bn().relu()    # (64, 48)
#         net = net.deconv(3, bias=True)      # (128, 96)
#
#     return tf.summary_image(net.sigmoid(), name='fake')


# @tf.scope
# def generator(z):
#     conv = dict(kernel=4, stride=1, padding='SAME')
#
#     with tf.default_args(conv=conv):
#         net = z
#         net = net.dense(4*3*512).bn().relu()
#         net = net.reshape((-1, 4, 3, 512))  # (4, 3)
#         net = net.sizeup(2).conv(256).bn().relu()   # (8, 6)
#         net = net.sizeup(2).conv(128).bn().relu()   # (16, 12)
#         net = net.sizeup(2).conv(64).bn().relu()    # (32, 24)
#         net = net.sizeup(2).conv(32).bn().relu()    # (64, 48)
#         net = net.sizeup(2).conv(3, bias=True)      # (128, 96)
#
#     return tf.summary_image(net.sigmoid(), name='fake')


@tf.scope
def generator(z):
    conv = dict(kernel=4, stride=1, padding='SAME')
    subpixel = dict(kernel=3, factor=2, padding='SAME', bias=True)

    with tf.default_args(conv=conv, subpixel=subpixel):
        net = z
        net = net.dense(4*3*512).bn().relu()
        net = net.reshape((-1, 4, 3, 512))  # (4, 3)
        net = net.subpixel().bn().relu().conv(256).bn().relu()   # (8, 6)
        net = net.subpixel().bn().relu().conv(128).bn().relu()   # (16, 12)
        net = net.subpixel().bn().relu().conv(64).bn().relu()    # (32, 24)
        net = net.subpixel().bn().relu().conv(32).bn().relu()    # (64, 48)
        net = net.subpixel().bn().relu().conv(3, bias=True)      # (128, 96)

    return tf.summary_image(net.sigmoid(), name='fake')


@tf.scope
def discriminator(x, zdim):
    conv = dict(kernel=3, padding='VALID', bias=True)

    with tf.default_args(conv=conv, dense=dict(bias=True)):
        net = x
        net = net.conv(16, stride=2).leaky(0.2).dropout(0.5)   # (63, 47)
        net = net.conv(32).leaky(0.2).dropout(0.5)             # (31, 23)
        net = net.conv(64, stride=2).leaky(0.2).dropout(0.5)   # (15, 11)
        net = net.conv(128).leaky(0.2).dropout(0.5)            # (7, 5)
        net = net.conv(256, stride=2).leaky(0.2).dropout(0.5)  # (3, 2)

        net = net.flat2d().dense(41 + zdim, bias=True)
        disc, klass, cont = tf.split(net, [1, 40, zdim], axis=1)

    return disc.squeeze(), klass, cont.sigmoid()


# @tf.scope
# def discriminator(x, zdim):
#     deconv = dict(kernel=3, padding='SAME', bias=True)
#
#     with tf.default_args(deconv=deconv, dense=dict(bias=True)):
#         net = x
#         net = net.sizedown(2).deconv(16).leaky(0.2).dropout(0.5)   # (64, 48)
#         net = net.sizedown(2).deconv(32).leaky(0.2).dropout(0.5)   # (32, 24)
#         net = net.sizedown(2).deconv(64).leaky(0.2).dropout(0.5)   # (16, 12)
#         net = net.sizedown(2).deconv(128).leaky(0.2).dropout(0.5)  # (8, 6)
#         net = net.sizedown(2).deconv(256).leaky(0.2).dropout(0.5)  # (4, 3)
#
#         net = net.flat2d().dense(41 + zdim, bias=True)
#         disc, klass, cont = tf.split(net, [1, 40, zdim], axis=1)
#
#     return disc.squeeze(), klass, cont.sigmoid()


# @tf.scope
# def discriminator(x, zdim):
#     atrous = dict(kernel=3, rate=3, padding='SAME', bias=True)
#
#     with tf.default_args(atrous=atrous, dense=dict(bias=True)):
#         net = x
#         net = net.sizedown(2).atrous(16).leaky(0.2).dropout(0.5)   # (64, 48)
#         net = net.sizedown(2).atrous(32).leaky(0.2).dropout(0.5)   # (32, 24)
#         net = net.sizedown(2).atrous(64).leaky(0.2).dropout(0.5)   # (16, 12)
#         net = net.sizedown(2).atrous(128).leaky(0.2).dropout(0.5)  # (8, 6)
#         net = net.sizedown(2).atrous(256).leaky(0.2).dropout(0.5)  # (4, 3)
#
#         net = net.flat2d().dense(41 + zdim, bias=True)
#         disc, klass, cont = tf.split(net, [1, 40, zdim], axis=1)
#
#     return disc.squeeze(), klass, cont.sigmoid()


# @tf.scope
# def discriminator(x, zdim):
#     conv = dict(kernel=3, padding='SAME', bias=True)
#     dwconv = dict(kernel=2, stride=2, padding='VALID', bias=True)
#
#     with tf.default_args(conv=conv, dwconv=dwconv, dense=dict(bias=True)):
#         net = x
#         net = net.dwconv().relu().conv(32).relu().dropout()   # (32, 24)
#         net = net.dwconv().relu().conv(64).relu().dropout()   # (16, 12)
#         net = net.dwconv().relu().conv(16).relu().dropout()   # (64, 48)
#         net = net.dwconv().relu().conv(128).relu().dropout()  # (8, 6)
#         net = net.dwconv().relu().conv(256).relu().dropout()  # (4, 3)
#
#         net = net.flat2d().dense(41 + zdim, bias=True)
#         disc, klass, cont = tf.split(net, [1, 40, zdim], axis=1)
#
#     return disc.squeeze(), klass, cont.sigmoid()


def model_celeb_gan():
    from sflow.data import celeba
    batch = 16
    zdim = 100

    # shape...
    data = celeba.attribute_trainset(batch=batch, size=(128, 96), threads=8)
    x = data.image  # input image
    x = tf.summary_image(x, name='real')
    y = tf.ones(data.batch)

    y_disc = tf.concat(0, [y, tf.zeros(batch)])

    z_klass = tf.random_uniform((batch, 40), 0., 1.).greater(0.5).to_float()  # data.label.to_float()
    z_cond = tf.random_uniform((batch, zdim), 0., 1.)
    z = tf.concat(1, [z_klass, z_cond])

    fake = generator(z)
    xx = tf.concat(0, [x, fake])

    disc, klass, cont = discriminator(xx, zdim)

    _, disc_fake = tf.split(disc, 2)
    _, cont = tf.split(cont, 2)

    target = tf.concat(0, [data.label, z_klass])

    loss_disc = tf.binary_cross_entropy(disc, y_disc).mean()
    loss_gen = tf.binary_cross_entropy(disc_fake, y).mean()
    loss_klass = tf.binary_cross_entropy(klass, target).mean()  # + 0.01 * tf.abs(cont - z_cond).mean()

    return loss_disc, loss_gen, loss_klass


def train():
    loss_disc, loss_gen, loss_klass = model_celeb_gan()

    tf.summary_loss(loss_disc, name='loss_disc')
    tf.summary_loss(loss_gen, name='loss_gen')
    tf.summary_loss(loss_klass, name='loss_klass')

    train_d = tf.optim.Adam(lr=0.0001, beta1=0.5).minimize(loss_disc + loss_klass, scope='discriminator')
    train_g = tf.optim.Adam(lr=0.001, beta1=0.5).minimize(loss_gen + loss_klass, scope='generator')

    writer = tf.summary_writer(logdir='train/face/celeb_sub_conv')
    sess = tf.get_default_session()
    saver = tf.saver()

    for ep, gstep in tf.trainstep(maxep=100, epochper=60000//16, savers=saver):
        loss_d = sess.run([loss_disc, train_d])[0]
        loss_g = sess.run([loss_gen, train_g])[0]
        print([ep, gstep, loss_d, loss_g])
        if gstep % 30 == 0:
            writer.add_summary(gstep, sess=sess)


if __name__ == '__main__':
    train()

