# -*- coding: utf-8 -*-
# import sflow.tf as tf
import sflow.core.logg as logg


def load_caffe(prototxt, caffemodel, *args, **kwargs):
    import caffe

    net = caffe.Net(prototxt, caffemodel, *args, **kwargs)

    return net


def load_caffe_params(prototxt, caffemodel):
    from snipy.concurrent import threaded

    th = threaded(load_caffe_params_impl, prototxt, caffemodel)
    return th.result


def load_caffe_params_impl(prototxt, caffemodel, *args, **kwargs):
    import caffe

    infos = []
    params = []

    net = load_caffe(prototxt, caffemodel, caffe.TEST, *args, **kwargs)
    for name, layer in net.layer_dict.items():
        tname = layer.type

        # if tname in ('Input',):
        #     logg.info('skip {}[{}]'.format(name, tname))
        #     continue

        desc = '{}[{}].'.format(name, tname)

        lparams = [b.data for b in layer.blobs]
        lparams, doc = _convert_weight(tname, lparams)
        if not lparams:
            logg.warn('skip {}', desc)
            continue

        for p, d in zip(lparams, doc):
            info = desc + d + '[{}]'.format(p.shape)
            logg.info(info)
            infos.append(info)

        params.extend(lparams)

    return params, infos


def _convert_weight(tname, lparams):
    if 'Convolution' in tname:
        w = lparams[0].transpose((2,3,1,0))
        b = lparams[1]
        assert len(lparams) == 2
        return [w, b], ['w', 'b']
    elif 'InnerProduct' in tname:
        w = lparams[0].transpose((1,0))
        b = lparams[1]
        assert len(lparams) == 2
        return [w, b], ['w', 'b']

    return lparams, map(str, range(len(lparams)))


def load_caffe_params2(prototxt, caffemodel, *args, **kwargs):
    import caffe

    net = load_caffe(prototxt, caffemodel, caffe.TEST, *args, **kwargs)
    # layer names
    layers = [name for name in net.params]
    types = [layer.type for layer in net.layers]

    # k = [l for l in net.layer_dict]

    infos = []
    params = []

    for layer in layers:
        weights = [blob.data for blob in net.params[layer]]
        params.extend(weights)
        infos.extend([layer + str(i+1) for i in range(len(net.params[layer]))])

    return params, infos


def get_weights(layer, caffe_params):
    weights = [blob.data for blob in caffe_params]
    return weights


if __name__ == '__main__':
    prototxt = '/train/zoo/vgg.face/vgg_face_caffe/VGG_FACE_deploy.prototxt'
    caffemodel = '/train/zoo/vgg.face/vgg_face_caffe/VGG_FACE.caffemodel'

    load_caffe_params(prototxt, caffemodel)

