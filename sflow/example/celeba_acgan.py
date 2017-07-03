# -*- coding: utf-8 -*-
import sflow.gpu1 as tf


@tf.scope
def generator(z):
    deconv = dict(kernel=5, stride=2, padding='SAME')

    with tf.default_args(deconv=deconv):
        net = z
        net = net.dense(4*3*512).bn().relu()
        net = net.reshape((-1, 4, 3, 512))
        net = net.deconv(256).bn().relu()
        net = net.deconv(128).bn().relu()
        net = net.deconv(64).bn().relu()
        net = net.deconv(32).bn().relu()
        net = net.deconv(3, bias=True)

    return tf.summary_image(net.sigmoid(), name='fake')


@tf.scope
def discriminator(x, zdim):
    conv = dict(kernel=3, padding='VALID', bias=True)

    with tf.default_args(conv=conv, dense=dict(bias=True)):
        net = x
        net = net.conv(16, stride=2).leaky(0.2).dropout(0.5)
        net = net.conv(32).leaky(0.2).dropout(0.5)
        net = net.conv(64, stride=2).leaky(0.2).dropout(0.5)
        net = net.conv(128).leaky(0.2).dropout(0.5)
        net = net.conv(256, stride=2).leaky(0.2).dropout(0.5)

        net = net.flat2d().dense(41 + zdim, bias=True)
        disc, klass, cont = tf.split(net, [1, 40, zdim], axis=1)

    return disc.squeeze(), klass, cont.sigmoid()


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
    loss_klass = tf.binary_cross_entropy(klass, target).mean()   # + 0.01*tf.mean(tf.square(cont - z_cond))

    return loss_disc, loss_gen, loss_klass


def train():
    loss_disc, loss_gen, loss_klass = model_celeb_gan()

    tf.summary_loss(loss_disc, name='loss_disc')
    tf.summary_loss(loss_gen, name='loss_gen')
    tf.summary_loss(loss_klass, name='loss_klass')

    train_d = tf.optim.Adam(lr=0.0001, beta1=0.5).minimize(loss_disc + loss_klass, scope='discriminator')
    train_g = tf.optim.Adam(lr=0.001, beta1=0.5).minimize(loss_gen + loss_klass, scope='generator')

    writer = tf.summary_writer(logdir='face32/log/celeb002')
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

