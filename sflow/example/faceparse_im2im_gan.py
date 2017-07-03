# -*- coding: utf-8 -*-
import sflow.gpu0 as tf

# generate face images
# from face parse data
# training by helen dataset
# note : https://arxiv.org/pdf/1611.07004v1.pdf
# when infering, also training_mode to True


@tf.scope
def generator(mask):
    conv = dict(kernel=4, stride=2, padding='SAME')
    leaky = dict(slope=0.2)

    with tf.default_args(conv=conv, leaky=leaky):
        skips = []  # contracts

        net = mask
        net = net.conv(64).keep(skips).leaky()
        net = net.conv(128).bn().keep(skips).leaky()
        net = net.conv(256).bn().keep(skips).leaky()
        net = net.conv(512).bn().keep(skips).leaky()
        net = net.conv(512).bn().keep(skips).leaky()
        net = net.conv(512).bn().keep(skips).leaky()
        net = net.conv(512).bn().keep(skips).leaky()
        net = net.conv(512).bn()

    deconv = dict(kernel=4, stride=2, padding='SAME')

    with tf.default_args(deconv=deconv, leaky=leaky):
        net = net.relu().deconv(512).bn().dropout()
        net = skips.pop().cat(net).relu().deconv(1024).bn().dropout()
        net = skips.pop().cat(net).relu().deconv(1024).bn().dropout()
        net = skips.pop().cat(net).relu().deconv(1024).bn()
        net = skips.pop().cat(net).relu().deconv(1024).bn()
        net = skips.pop().cat(net).relu().deconv(512).bn()
        net = skips.pop().cat(net).relu().deconv(256).bn()
        net = skips.pop().cat(net).relu().deconv(128).bn()

        # net = tf.tanh(net).conv(3, kernel=1, stride=1, bias=True).sigmoid()
        net = tf.tanh(net).conv(3, kernel=1, stride=1, bias=True)
        net = tf.clip_by_value((net.sigmoid() - 0.5)*1.02 + 0.5, 0., 1.)

    return tf.summary_image(net, name='fake')


@tf.scope
def discriminator(x):

    conv = dict(kernel=4, stride=2, padding='SAME', bias=True)
    leaky = dict(slope=0.2)

    with tf.default_args(conv=conv, leaky=leaky, dense=dict(bias=True)):
        net = x
        net = net.conv(64).leaky()
        net = net.conv(128).bn().leaky()
        net = net.conv(256).bn().leaky()
        net = net.conv(512).bn().leaky()
        net = net.conv(1)

    return net


def model_im2im_gan():
    from sflow.data import helen
    batch = 16

    data = helen.trainset(batch=batch, size=(256, 256), threads=8, removebg=True)
    x = data.image  # input image

    tf.summary_image(x, name='real')

    fake = generator(data.label)

    realp = tf.concat(3, [data.label, data.image])  # real pair
    fakep = tf.concat(3, [data.label, fake])  # fake pair
    xx = tf.concat(0, [realp, fakep])   # all pair input to discriminator

    disc = discriminator(xx)

    _, disc_fake = tf.split(disc, 2)

    y_disc = tf.concat(0, [tf.ones(disc_fake.dims), tf.zeros(disc_fake.dims)])
    y_fake = tf.ones(disc_fake.dims)

    loss_disc = tf.binary_cross_entropy(disc, y_disc).mean()
    loss_gen = tf.binary_cross_entropy(disc_fake, y_fake).mean()

    loss_l1 = tf.abs(fake - data.image).mean()
    # loss_tv = tf.total_variance_iso(fake)

    return loss_disc, loss_gen, loss_l1


def train():
    loss_disc, loss_gen, loss_l1 = model_im2im_gan()

    tf.summary_loss(loss_disc, name='loss_disc')
    tf.summary_loss(loss_gen, name='loss_gen')
    tf.summary_loss(loss_l1, name='loss_l1')

    # decay = tf.decay.exponential(100, 0.99)
    train_d = tf.optim.Adam(lr=0.0001, beta1=0.5).minimize(loss_disc, scope='discriminator')
    # train_g = tf.optim.Adam(lr=0.001, beta1=0.5).minimize(loss_gen + 0.001 * loss_l1, scope='generator')
    train_g = tf.optim.Adam(lr=0.001, beta1=0.5).minimize(loss_gen + 0.001 * loss_l1, scope='generator')

    logdir = 'train/im2im/log/face.unet.fix.noz.bg.sigmoid'
    writer = tf.summary_writer(logdir=logdir)
    saver = tf.saver(savedir=logdir)

    sess = tf.get_default_session()

    for ep, gstep in tf.trainstep(maxep=10, epochper=60000//16, savers=saver):
        try:
            loss_d = sess.run([loss_disc, train_d])[0]
            loss_g, loss_l = sess.run([loss_gen, loss_l1, train_g])[:-1]

            print([ep, gstep, loss_d, loss_g, loss_l])
            if gstep % 30 == 0:
                writer.add_summary(gstep, sess=sess)
        except KeyboardInterrupt:
            tf.save_on_interrupt(saver, sess, ep)


if __name__ == '__main__':
    train()

