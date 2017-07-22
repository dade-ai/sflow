# -*- coding: utf-8 -*-
import sflow.tf as tf


@tf.scope
def generator(z):

    deconv = dict(kernel=4, stride=2, padding='SAME')

    with tf.default_args(deconv=deconv):
        net = z
        net = net.dense(1024).bn().pleaky()
        net = net.dense(7*7*128).bn().pleaky()
        net = net.reshape((-1, 7, 7, 128))
        net = net.deconv(64).bn().pleaky()
        net = net.deconv(1).sigmoid()

        return tf.summary_image(net, name='fake')


@tf.scope
def discriminator(imgz):

    conv = dict(kernel=4, stride=2, padding='SAME')

    with tf.default_args(conv=conv):
        net = imgz
        net = net.conv(64).leaky()
        net = net.conv(128).leaky().flat2d()
        net = net.dense(1024).leaky()
        net = net.dense(1).squeeze()
        # prob = net.sigmoid()
        # must return logits

        return tf.identity(net, name='disc')


def model_mnist_gan():
    from sflow.data.mnist import dataset_train

    # MNIST input tensor ( with QueueRunner )
    batch = 32
    data = dataset_train(batch=batch)
    data.batch = batch
    x = data.image  # input image
    y = tf.ones(data.batch)

    ydisc = tf.concat(0, [y, y*0.])
    z = tf.random_uniform((data.batch, 100))

    fake = generator(z)
    xx = tf.concat(0, [x, fake])

    disc = discriminator(xx)
    _, disc_fake = tf.split(disc, 2)

    loss_disc = tf.binary_cross_entropy(disc, ydisc).mean()
    loss_gen = tf.binary_cross_entropy(disc_fake, y).mean()

    return loss_disc, loss_gen


def train():
    loss_disc, loss_gen = model_mnist_gan()
    losses = [loss_disc, loss_gen]

    tf.summary_loss(loss_disc, name='loss_disc')
    tf.summary_loss(loss_gen, name='loss_gen')

    decay = tf.decay.exponential(100, 0.99)
    tf.optim.MaxProp(lr=decay(0.0001)).minimize(loss_disc, scope='discriminator')
    tf.optim.MaxProp(lr=decay(0.001)).minimize(loss_gen, scope='generator')

    writer = tf.summary_writer(logdir='train/log/test')
    sess = tf.get_default_session()
    saver = tf.saver()
    for ep, gstep, losses in tf.trainall(losses, saver, maxep=100, epochper=60000//16):
        print([ep, gstep, losses])
        if gstep % 100 == 0:
            writer.add_summary(gstep, sess=sess)


if __name__ == '__main__':
    train()

