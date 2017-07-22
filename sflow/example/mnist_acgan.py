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
def discriminator(xx, num_cont, batch):

    conv = dict(kernel=4, stride=2, bias=True, padding='SAME')

    with tf.default_args(conv=conv):
        net = xx
        net = net.conv(64).leaky()
        net = net.conv(128).leaky().flat2d()
        net = net.dense(1024).leaky()

        disc = net.dense(1).squeeze()
        net = net.dense(128).leaky()
        klass = net.dense(10)
        cont = net[batch:].dense(num_cont).sigmoid()

    return disc, klass, cont


def model_acgan(batch, num_dim, num_cont):
    from sflow.data.mnist import dataset_train

    # MNIST input tensor ( with QueueRunner )
    data = dataset_train(batch=batch)

    x = data.image  # input image
    y = tf.ones(batch)
    y_disc = tf.concat(0, [y, tf.zeros(batch)])

    z_cat = tf.multinomial(tf.ones((batch, 10)), 1).to_int32().squeeze()
    z = z_cat.one_hot(10).hcat(tf.random_uniform((batch, num_dim - 10)))
    z_cont = z[:, 10:10+num_cont]

    label = tf.concat(0, [data.label, z_cat])

    gen = generator(z)
    xx = tf.concat(0, [x, gen])
    disc, klass, cont = discriminator(xx, num_cont, batch)

    loss_disc = tf.binary_cross_entropy(disc, y_disc).mean()
    loss_gen = tf.binary_cross_entropy(disc[batch:], y).mean()
    loss_klass = tf.softmax_cross_entropy(klass, label).mean() + tf.nn.l2_loss(cont - z_cont)

    tf.summary_loss(loss_disc, name='loss_disc')
    tf.summary_loss(loss_gen, name='loss_gen')
    tf.summary_loss(loss_klass, name='loss_klass')

    return loss_disc, loss_gen, loss_klass


def train():
    batch = 32
    num_dim = 50
    num_cont = 2

    loss_disc, loss_gen, loss_klass = model_acgan(batch, num_dim, num_cont)

    decay = tf.decay.exponential(100, 0.99)
    tf.optim.Adam(lr=decay(0.0001)).minimize(loss_disc + loss_klass, scope='discriminator')
    tf.optim.Adam(lr=decay(0.001)).minimize(loss_gen + loss_klass, scope='generator')

    writer = tf.summary_writer(logdir='acgan/log/test')
    sess = tf.get_default_session()
    losses = [loss_disc, loss_gen, loss_klass]
    for ep, gstep, losses in tf.trainall(losses, tf.saver(), maxep=10, epochper=60000//16):
        print(ep, gstep, losses)
        if gstep % 100 == 0:
            writer.add_summary(gstep, sess=sess)


if __name__ == '__main__':
    train()


