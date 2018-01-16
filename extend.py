import mxnet as mx
import numpy as np


class MDNetMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(MDNetMetric, self).__init__('MDNetAcc')
        self.pred, self.label = ['score'], ['label']

    def update(self, labels, preds):
        # ctx = mx.cpu(0)
        ctx = mx.gpu(1)
        label = labels[0].reshape((-1,)).as_in_context(ctx)
        pred = preds[0].as_in_context(ctx)
        loss = mx.ndarray.softmax_cross_entropy(pred, label).asnumpy()
        # print loss
        label = label.asnumpy()
        pred = pred.asnumpy()
        true_num = np.sum(pred.argmax(1) == label)
        false_num = label.shape[0] - true_num
        # print 'success:%d,fail:%d' % (true_num, false_num)

        self.sum_metric += true_num
        self.num_inst += label.shape[0]
