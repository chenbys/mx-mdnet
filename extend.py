import mxnet as mx
import numpy as np
from setting import config


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
        self.acc_th = acc_th

    def update(self, labels, preds):
        label = labels[0].asnumpy().reshape((-1,))
        pred = preds[0].asnumpy()
        acc = np.sum(abs(label - pred) < self.acc_th)
        self.sum_metric += acc * 100
        self.num_inst += len(label)


class MDNetIOULoss(mx.metric.EvalMetric):
    def __init__(self):
        super(MDNetIOULoss, self).__init__('MDNetIOULoss')

    def update(self, labels, preds):
        label = labels[0].reshape((-1,)).as_in_context(config.ctx)
        pred = preds[0].as_in_context(config.ctx)
        loss = mx.ndarray.smooth_l1(pred - label, scalar=1).asnumpy().sum()
        self.sum_metric += loss * 1000
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
