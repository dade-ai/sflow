# -*- coding: utf-8 -*-
import sflow.gpu0 as tf

# http://arxiv.org/pdf/1312.6114v10.pdf


@tf.scope
def encoder(x, ndim):

    conv = dict(kernel=4, padding='SAME')

    with tf.default_args(conv=conv):
        net = x
        net = net.conv(64).bn().relu()
        net = net.conv(64, stride=2).bn().relu()
        net = net.conv(128).bn().relu()
        net = net.conv(128, stride=2).bn().relu()
        net = net.conv(256).bn().relu()
        net = net.conv(256, stride=2).bn().relu()
        net = net.gpool().conv(ndim*2, kernel=1, bias=True)

    return net


@tf.scope
def decoder(z, outdim):

    deconv = dict(kernel=4, padding='SAME')

    with tf.default_args(deconv=deconv):
        net = z
        net = net.deconv(256, kernel=7, padding='VALID').bn().relu()
        net = net.deconv(256).bn().relu()
        net = net.deconv(256).bn().relu()
        net = net.deconv(128, stride=2).bn().relu()
        net = net.deconv(128).bn().relu()
        net = net.deconv(64, stride=2).bn().relu()
        net = net.deconv(64).bn().relu()
        net = net.deconv(outdim, bias=True)

    return net


@tf.scope
def reparameterize(z):
    m, s = z.split(axis=-1, count=2)
    #     v = s.exp()  # enforce positive, translate as a std**2
    v = tf.nn.softplus(s) + 1e-8  # enforce positive, translate as a std**2
    return zsampling(m, v), m, v


@tf.scope
def zsampling(m=None, v=None, dims=None):

    # equation(10) mu + sigma*eps
    # half dimension
    if v is None:
        # return jt.rand.normal(dims)
        return m + tf.random_normal(dims)
    else:
        dims = dims or m.dims
        eps = tf.random_normal(dims)
        std = v.sqrt()
        return m + std * eps


@tf.scope
def vae_cost(m, v, eps=1e-8):
    # appendix(B) (11page) D_kl
    # -sum(0.5 * (1.0 + log(std^2) - m^2 - std^2))

    return -0.5 * tf.sum(1.0 + tf.log(v + eps) - m.square() - v)


@tf.scope
def reconstruction_cost(y, x, eps=1e-8):
    """Reconstruction loss
    Cross entropy reconstruction loss
    Args:
        y: tensor produces by decoder
        x: the target tensor that we want to reconstruct
        eps:
    """

    # data must be in (0, 1)
    # binary cross entropy
    # y = y.sigmoid()
    # return tf.sum(-x * tf.log(y + eps) - (1.0 - x) * tf.log(1.0 - y + eps))
    # odd
    return tf.binary_cross_entropy(y, x).sum()


def model_vae(x, ndim=2):

    z = encoder(x, ndim)
    z, m, v = reparameterize(z)

    # reconstructed
    r = decoder(z, x.dims[-1])  # reconstructed (logit)
    img = r.sigmoid()  # reconstructed image
    tf.summary_activation(r, 'image_logit')
    tf.summary_activation(img, 'image')
    tf.summary_image(r.sigmoid(), 'reconstruct')

    loss_r = reconstruction_cost(r, x)
    tf.summary.histogram('z_m', m)
    tf.summary.histogram('z_v', v)
    loss_vae = vae_cost(m, v)

    return loss_r, loss_vae


def train():
    from sflow.data.mnist import dataset_train
    data = dataset_train(batch=16)
    x = data.image  # input image
    tf.summary_image(x, name='org')

    zdim = 2
    loss_r, loss_vae = model_vae(x, zdim)

    tf.summary_loss(loss_r, name='recon')
    tf.summary_loss(loss_vae, name='vae')

    decay = tf.decay.exponential(1000, 0.99)
    tf.optim.Adam(lr=decay(0.001)).minimize(loss_r + loss_vae)

    logdir = '/train/vae/mnist.001'
    saver = tf.saver(savedir=logdir)
    writer = tf.summary_writer(logdir=logdir)
    sess = tf.get_default_session()

    for ep, gstep, loss in tf.trainall([loss_r, loss_vae], ep=10,
                                       savers=saver, epochper=60000//16):
        print([ep, gstep, loss])
        if gstep % 30 == 0:
            writer.add_summary(gstep, sess=sess)


if __name__ == '__main__':
    train()

