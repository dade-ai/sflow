# -*- coding: utf-8 -*-
import sflow.tf as tf
from sflow.data import helen

# face parse from helen


@tf.scope(name='face_parse')
def face_parse(image, label):

    conv = dict(kernel=3, padding='SAME')
    deconv = dict(kernel=3, padding='SAME', bias=True)
    subpixel = dict(kernel=3, factor=2, padding='SAME')

    with tf.default_args(conv=conv, deconv=deconv, subpixel=subpixel):
        net = image
        net = net.conv(16).bn().relu().conv(16).bn().relu().maxpool()
        net = net.conv(32).bn().relu().conv(32).bn().relu().maxpool()
        net = net.conv(64).bn().relu().conv(64).bn().relu().maxpool()
        net = net.conv(64).bn().relu().conv(64).bn().relu().maxpool()
        net = net.conv(64).bn().relu()
        net = net.deconv(64).bn().relu()
        net = net.subpixel().deconv(64).bn().relu().deconv(64).bn().relu()
        net = net.subpixel().deconv(64).bn().relu().deconv(64).bn().relu()
        net = net.subpixel().deconv(32).bn().relu().deconv(32).bn().relu()
        net = net.subpixel().deconv(16).bn().relu().deconv(11, bias=True)

        prob = net.softmax()
        summary_parse(prob)

        losses = tf.softmax_cross_entropy(net, label, name='losses')
        loss = losses.mean()

    return tf.dic(losses=losses, loss=loss, outputs=[prob])


@tf.scope(name='face_parse')
def face_parse2(image, label):

    conv = dict(kernel=3, padding='SAME')
    deconv = dict(kernel=3, padding='SAME', bias=True)
    # subpixel = dict(kernel=3, factor=2, padding='SAME')
    maxpool_where = dict(kernel=2)
    unpool_where = dict(kernel=2)

    with tf.default_args(conv=conv, deconv=deconv,
                         maxpool_where=maxpool_where,
                         unpool_where=unpool_where):
        net = image
        wheres = []
        net, where = net.conv(16).bn().relu().conv(16).bn().relu().maxpool_where()
        wheres.append(where)
        net, where = net.conv(32).bn().relu().conv(32).bn().relu().maxpool_where()
        wheres.append(where)
        net, where = net.conv(64).bn().relu().conv(64).bn().relu().maxpool_where()
        wheres.append(where)
        net, where = net.conv(64).bn().relu().conv(64).bn().relu().maxpool_where()
        wheres.append(where)
        net = net.conv(64).bn().relu()
        net = net.deconv(64).bn().relu()
        net = net.unpool_where(wheres.pop()).deconv(64).bn().relu().deconv(64).bn().relu()
        net = net.unpool_where(wheres.pop()).deconv(64).bn().relu().deconv(32).bn().relu()
        net = net.unpool_where(wheres.pop()).deconv(32).bn().relu().deconv(16).bn().relu()
        net = net.unpool_where(wheres.pop()).deconv(16).bn().relu().deconv(11, bias=True)

        net = tf.summary_activation(net, name='logits')
        prob = net.softmax()

        summary_parse(prob)

        # losses = tf.nn.sigmoid_cross_entropy_with_logits(net, label)
        losses = tf.softmax_cross_entropy(net, label, name='losses')
        loss = losses.mean()

    return tf.dic(loss=loss, logits=net, label=label, image=image)


def summary_parse(prob):
    tf.summary.image('prob0', prob[:, :, :, :3])
    tf.summary.image('prob1', prob[:, :, :, 3:6])
    tf.summary.image('prob2', prob[:, :, :, 6:9])
    tf.summary.image('prob3', prob[:, :, :, 9].expand_dims(-1))
    tf.summary.image('prob4', prob[:, :, :, 10].expand_dims(-1))


def train_test():
    # model build
    data = helen.trainset(batch=16, threads=8)
    model = face_parse(data.image, data.label)

    optim = tf.train.AdamOptimizer(learning_rate=0.001)
    trainop = optim.minimize(model.loss)  # train op

    tf.summary.scalar('loss', model.loss)
    # tf.summary_loss(model.losses)

    summary = tf.summary.merge_all()
    twriter = tf.summary.FileWriter('/tmp/face_parse/005')

    tf.global_variables_initializer().run()

    with tf.feeding() as (sess, coord):
        i = 0
        while True:
            s, loss, _ = sess.run([summary, model.loss, trainop])
            # loss = sess.run([model.loss, trainop])
            print(loss)
            twriter.add_summary(s, i)
            i += 1


def train():
    # model build
    data = helen.trainset(batch=16, threads=8)
    model = face_parse2(data.image, data.label)

    # savepath = '/data/train/face_parse/test/face'
    tf.train.AdamOptimizer(0.001).minimize(model.loss)

    writer = tf.summary_writer('/data/train/face/faceparse/log/005')

    sess = tf.get_default_session()
    for (ep, gstep, out) in tf.trainall([model.loss], epochper=10000):
        if gstep % 100 == 0:
            print(ep, gstep, out)
            writer.add_summary(gstep, sess=sess)


if __name__ == '__main__':
    train()

