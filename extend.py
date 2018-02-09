import mxnet as mx
import numpy as np
from setting import config
import csym


class MDNetACC(mx.metric.EvalMetric):
    def __init__(self):
        super(MDNetACC, self).__init__('MDNetAcc')
        self.pred, self.label = ['score'], ['label']

    def update(self, labels, preds):
        label = labels[0].reshape((-1,)).asnumpy()
        pred = preds[0].asnumpy()
        true_num = np.sum(pred.argmax(1) == label)
        self.sum_metric += true_num
        self.num_inst += label.shape[0]


class MDNetLoss(mx.metric.EvalMetric):
    def __init__(self):
        super(MDNetLoss, self).__init__('MDNetLoss')
        self.pred, self.label = ['score'], ['label']

    def update(self, labels, preds):
        label = labels[0].reshape((-1,)).as_in_context(config.ctx)
        pred = preds[0].as_in_context(config.ctx)
        loss = mx.ndarray.softmax_cross_entropy(pred, label).asnumpy()
        if loss > 7000:
            print pred.asnumpy()
            exit(0)
        self.sum_metric += loss
        self.num_inst += label.shape[0]


class MDNetIOUACC(mx.metric.EvalMetric):
    def __init__(self, acc_th=0.1):
        super(MDNetIOUACC, self).__init__('MDNetIOUACC_' + str(acc_th))
        self.acc_th = acc_th * acc_th / 2.

    def update(self, labels, preds):
        label = labels[0].asnumpy().reshape((-1,))
        pred = preds[0].asnumpy()
        acc = np.sum(pred < self.acc_th)
        self.sum_metric += acc * 100
        self.num_inst += len(label)


# class MDNetIOUACC_(mx.metric.EvalMetric):
#     def __init__(self, acc_th=0.1):
#         super(MDNetIOUACC_, self).__init__('MDNetIOUACC__' + str(acc_th))
#         self.acc_th = acc_th
#
#     def update(self, labels, preds):
#         label = labels[0].asnumpy().reshape((-1,))
#         pred = preds[0].asnumpy()
#         acc = np.sum(abs(label - pred) < self.acc_th)
#         self.sum_metric += acc * 100
#         self.num_inst += len(label)


class MDNetIOULoss(mx.metric.EvalMetric):
    def __init__(self):
        super(MDNetIOULoss, self).__init__('MDNetIOULoss')

    def update(self, labels, preds):
        label = labels[0].reshape((-1,)).as_in_context(config.ctx)
        pred = preds[0].as_in_context(config.ctx)
        loss = mx.ndarray.smooth_l1(pred - label, scalar=1).asnumpy().sum()
        self.sum_metric += pred.sum().asnumpy() * 1000
        self.num_inst += len(label)


def get_mdnet_conv123_params(prefix='', mat_path='saved/conv123.mat'):
    import scipy.io as sio
    import numpy as np
    conv123 = sio.loadmat(mat_path)
    conv123 = conv123['conv123']
    conv1_weight = conv123[0, 0][0]
    conv1_bias = conv123[0, 0][1]
    conv2_weight = conv123[0, 0][2]
    conv2_bias = conv123[0, 0][3]
    conv3_weight = conv123[0, 0][4]
    conv3_bias = conv123[0, 0][5]

    conv1_weight_ = np.transpose(conv1_weight, [3, 2, 0, 1])
    conv1_bias_ = conv1_bias.reshape((96,))
    conv2_weight_ = np.transpose(conv2_weight, [3, 2, 0, 1])
    conv2_bias_ = conv2_bias.reshape((256,))
    conv3_weight_ = np.transpose(conv3_weight, [3, 2, 0, 1])
    conv3_bias_ = conv3_bias.reshape((512,))

    arg_params = dict()
    arg_params[prefix + 'conv1_weight'] = conv1_weight_
    arg_params[prefix + 'conv1_bias'] = conv1_bias_
    arg_params[prefix + 'conv2_weight'] = conv2_weight_
    arg_params[prefix + 'conv2_bias'] = conv2_bias_
    arg_params[prefix + 'conv3_weight'] = conv3_weight_
    arg_params[prefix + 'conv3_bias'] = conv3_bias_

    return arg_params


def init_model(loss_type=0, fixed_conv=0, saved_fname='conv123'):
    import datahelper

    if loss_type == 0:
        sym = csym.get_mdnet()
    elif loss_type == 1:
        sym = csym.get_mdnet_with_smooth_l1_loss()
    fixed_param_names = []
    for i in range(1, fixed_conv + 1):
        fixed_param_names.append('conv' + str(i) + '_weight')
        fixed_param_names.append('conv' + str(i) + '_bias')
    model = mx.mod.Module(symbol=sym, context=config.ctx, data_names=('image_patch', 'feat_bbox',),
                          label_names=('label',),
                          fixed_param_names=fixed_param_names)
    sample_iter = datahelper.get_train_iter(
        datahelper.get_train_data('saved/mx-mdnet_01CE.jpg', [24, 24, 24, 24], iou_label=bool(loss_type)))
    model.bind(sample_iter.provide_data, sample_iter.provide_label)

    all_params = {}
    if saved_fname == 'conv123':
        print '@CHEN->load params from conv123'
        conv123 = get_mdnet_conv123_params()
        for k in conv123.keys():
            conv123[k] = mx.ndarray.array(conv123.get(k))
        model.init_params(arg_params=conv123, allow_missing=True, force_init=False, allow_extra=True)
    elif saved_fname is not None:
        print '@CHEN->load all params from:' + saved_fname
        all_params, arg_params = load_all_params(saved_fname)
        model.set_params(arg_params, None)
    else:
        print '@CHEN->init params.'
        model.init_params()

    return model, all_params


def get_MD_params(seq_name, arg_params, all_params):
    branch_params = all_params.get(seq_name)
    if branch_params is None:
        print 'branch_params is None'
        return arg_params

    shared_params_names = ['conv1_weight', 'conv1_bias', 'conv2_weight', 'conv2_bias', 'conv3_weight', 'conv3_bias']
    shared_params = {}
    for name in shared_params_names:
        shared_params[name] = arg_params[name]

    branch_params.update(shared_params)
    return branch_params


def save_all_params(all_params, k):
    import os
    os.system('mkdir params/' + str(k))
    for key, value in all_params.items():
        mx.ndarray.save(os.path.join('params', str(k), key), value)


def load_all_params(fname):
    k = int(fname)
    all_params = mx.ndarray.load(fname)
    return all_params, all_params.get(all_params.keys()[-1])
