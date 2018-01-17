import mxnet as mx
import numpy as np
from setting import config
from kit import p


class MDNetMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(MDNetMetric, self).__init__('MDNetAcc')
        self.pred, self.label = ['score'], ['label']

    def update(self, labels, preds):
        label = labels[0].reshape((-1,)).as_in_context(config.ctx)
        pred = preds[0].as_in_context(config.ctx)
        loss = mx.ndarray.softmax_cross_entropy(pred, label).asnumpy()
        p(loss)
        label = label.asnumpy()
        pred = pred.asnumpy()
        true_num = np.sum(pred.argmax(1) == label)
        false_num = label.shape[0] - true_num
        p('success:%d,fail:%d' % (true_num, false_num))

        self.sum_metric += true_num
        self.num_inst += label.shape[0]


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
