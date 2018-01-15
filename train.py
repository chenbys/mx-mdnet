import mxnet as mx
import numpy as np
from symbol import mdnet
import cv2
import matplotlib.pyplot as plt
import sample
import logging
import argparse
import os
from dataset.OTBHelper import OTBHelper


def train_on_one_frame(args, image_path, region):
    '''

    :param image_path:
    :param region:
    :return:
    '''
    image = cv2.imread(image_path)
    train_data = sample.train_data(image, region)
    train_iter = get_train_iter(train_data)
    # train_iter = MDNetIter(image, region)

    image_patch = mx.symbol.Variable(name='image_patch')
    feat_bbox = mx.symbol.Variable(name='feat_bbox')
    label = mx.symbol.Variable(name='label')
    sym = mdnet.get_mdnet(image_patch, feat_bbox, label)
    if args.gpu == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu)

    model = mx.mod.Module(symbol=sym, context=ctx, data_names=('image_patch', 'feat_bbox',),
                          label_names=('label',))

    # model.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    # model.init_params()
    # one batch: prob:(1521,2),label:(1,1521)
    eval_metrics = mx.metric.CompositeEvalMetric()
    eval_metrics.add(MDNetMetric())
    logging.getLogger().setLevel(logging.DEBUG)
    model.fit(train_iter, optimizer='sgd',
              optimizer_params={'learning_rate': args.lr,
                                },
              eval_metric=eval_metrics, num_epoch=args.num_epoch, batch_end_callback=mx.callback.Speedometer(1))
    print 'over'


def test(model):
    pass


def get_train_iter(train_data):
    image_patches = list()
    feat_bboxes = list()
    labels = list()
    for item in train_data:
        image_patch, feat_bbox, label = item
        image_patches.append(image_patch)
        feat_bboxes.append(feat_bbox)
        labels.append(label)

    return mx.io.NDArrayIter({'image_patch': image_patches, 'feat_bbox': feat_bboxes}, {'label': labels},
                             batch_size=1, data_name=('image_patch', 'feat_bbox',), label_name=('label',))


class MDNetIter(mx.io.DataIter):
    def __init__(self, image, region):
        self.batch_size = 1
        self._provide_data = zip(('image_patch', 'feat_bbox',),
                                 ((1, 3, 227, 227), (1521, 5)))
        self._provide_label = zip(('label',), ((1521,)))
        self.train_data = sample.train_data(image, region)
        self.num_batches = len(self.train_data)
        self.cur_batch = 0

    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        if self.cur_batch < self.num_batches:
            image_patch = self.train_data[self.cur_batch][0]
            feat_bbox = self.train_data[self.cur_batch][1]
            label = self.train_data[self.cur_batch][2]
            self.cur_batch += 1
            return mx.io.DataBatch([image_patch, feat_bbox], [label])
        else:
            raise StopIteration


class MDNetMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(MDNetMetric, self).__init__('MDNetAcc')
        self.pred, self.label = ['score'], ['label']

    def update(self, labels, preds):
        label = labels[0].reshape((-1,)).as_in_context(mx.gpu(1))
        pred = preds[0].as_in_context(mx.gpu(1))
        label_np = label.asnumpy()
        pred_np = pred.asnumpy()
        loss = mx.ndarray.softmax_cross_entropy(pred, label).asnumpy()
        print loss

        true_num = np.sum(pred_np.argmax(1) == label_np)
        false_num = label_np.shape[0] - true_num
        # print 'success:%d,fail:%d' % (true_num, false_num)

        self.sum_metric += true_num
        self.num_inst += label_np.shape[0]


def parse_args():
    parser = argparse.ArgumentParser(description='Train MDNet network')
    parser.add_argument('--gpu', help='GPU device to train with', default=-1, type=int)
    parser.add_argument('--num_epoch', help='epoch of training', default=5, type=int)
    parser.add_argument('--lr', help='base learning rate', default=0.001, type=float)
    parser.add_argument('--wd', help='base learning rate', default=0.005, type=float)
    parser.add_argument('--OTB_path', help='OTB folder', default='/Users/chenjunjie/workspace/OTB', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    seq_name = 'Sufer'
    otb = OTBHelper(args.OTB_path)
    img_list = otb.get_img(seq_name)
    gt_list = otb.get_gt(seq_name)
    for img_path, gt in zip(img_list, gt_list):
        train_on_one_frame(args, img_path, gt)


if __name__ == '__main__':
    main()
