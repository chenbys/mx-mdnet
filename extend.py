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


def load_net(prefix='', mat_path='saved/conv123.mat'):
    import scipy.io as sio
    conv123 = sio.loadmat(mat_path)
    conv123 = conv123['conv123']
    conv1_filters = conv123[0, 0][0]
    conv1_biases = conv123[0, 0][1]
    conv2_filters = conv123[0, 0][2]
    conv2_biases = conv123[0, 0][3]
    conv3_filters = conv123[0, 0][4]
    conv3_biases = conv123[0, 0][5]
    arg_params = dict()
    # arg_params[prefix + 'conv1'] = conv1
    # arg_params[prefix + 'conv2'] = conv2
    # arg_params[prefix + 'conv3'] = conv3

    return arg_params
