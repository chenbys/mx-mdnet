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


def train_on_one_frame(args, image_path, region, model=None, begin_epoch=0, num_epoch=50, ctx=mx.gpu(1), val_iter=None):
    '''

    :param image_path:
    :param region:
    :return:
    '''
    image = cv2.imread(image_path)
    train_iter = get_train_iter(image, region)

    if model is None:
        image_patch = mx.symbol.Variable(name='image_patch')
        feat_bbox = mx.symbol.Variable(name='feat_bbox')
        label = mx.symbol.Variable(name='label')
        sym = mdnet.get_mdnet(image_patch, feat_bbox, label)
        model = mx.mod.Module(symbol=sym, context=ctx, data_names=('image_patch', 'feat_bbox',),
                              label_names=('label',))

    eval_metrics = mx.metric.CompositeEvalMetric()
    eval_metrics.add(MDNetMetric())
    logging.getLogger().setLevel(logging.DEBUG)
    model.fit(train_data=train_iter, eval_data=val_iter, optimizer='sgd',
              optimizer_params={'learning_rate': args.lr,
                                'wd'           : args.wd},
              eval_metric=eval_metrics, num_epoch=begin_epoch+num_epoch, begin_epoch=begin_epoch,
              batch_end_callback=mx.callback.Speedometer(1))
    print 'over'
    return model


def test(model, img_path, pre_region):
    img = cv2.imread(img_path)
    feat_bbox = sample.sample_on_feat()
    predict_iter = get_predict_iter(img, pre_region, feat_bbox)
    # res.shape=(2304,2)
    res = model.predict(predict_iter)
    pos_score = res[:, 1]
    opt_idx = mx.ndarray.topk(pos_score, k=5).asnumpy().astype('int32')
    opt_feat_bboxes = feat_bbox[opt_idx, 1:]
    opt_img_bboxes = sample.feat2img(opt_feat_bboxes)
    opt_img_bbox = opt_img_bboxes.mean(0)
    return opt_img_bbox


def get_train_iter(img, region, stride_w=0.2, stride_h=0.2):
    from scipy.misc import imresize
    import util
    img_H, img_W, c = np.shape(img)
    x, y, w, h = region
    X, Y, W, H = x - w / 2., y - h / 2., 2 * w, 2 * h
    patches = list()
    for scale_w in np.arange(0.5, 1.6, stride_w):
        for scale_h in np.arange(0.5, 1.6, stride_h):
            W_, H_ = W * scale_w, H * scale_h
            X_, Y_ = x + w / 2. - W_ / 2., y + h / 2. - H_ / 2.
            # in case of out of range
            X_, Y_ = max(0, X_), max(0, Y_)
            W_, H_ = min(img_W - X_, W_), min(img_H - Y_, H_)
            patches.append([X_, Y_, W_, H_])

    image_patches = list()
    feat_bboxes = list()
    labels = list()
    for patch in patches:
        # crop image as train_data
        X, Y, W, H = patch
        img_patch = imresize(img[int(Y):int(Y + H), int(X):int(X + W), :], [227, 227])
        # ISSUE: change HWC to CHW
        img_patch = img_patch.reshape((3, 227, 227))

        # get region
        label_region = np.array([[227. * (x - X) / W, 227. * (y - Y) / H, 227. * w / W, 227. * h / H]])
        # get 50 pos samples
        # feat_boxes = sample()
        # img_boxes = feat2img(feat_boxes)
        label_feat = sample.x1y2x2y22xywh(sample.img2feat(sample.xywh2x1y1x2y2(label_region)))
        feat_bbox, label = sample.get_samples(label_feat)
        # get train_label
        image_patches.append(img_patch)
        feat_bboxes.append(feat_bbox)
        labels.append(label)

    return mx.io.NDArrayIter({'image_patch': image_patches, 'feat_bbox': feat_bboxes}, {'label': labels},
                             batch_size=1, data_name=('image_patch', 'feat_bbox',), label_name=('label',))


def get_predict_iter(img, pre_region, feat_bbox):
    from scipy.misc import imresize
    x, y, w, h = pre_region
    img_H, img_W, c = np.shape(img)
    img_pad = np.concatenate((img, img, img), 0)
    img_pad = np.concatenate((img_pad, img_pad, img_pad), 1)
    W, H = 227 / 131. * w, 227 / 131. * h
    X, Y = img_W + x + w / 2. - W / 2., img_H + y + h / 2. - H / 2.
    img_patch = img_pad[int(Y):int(Y + H), int(X):int(X + W), :]
    img_patch = imresize(img_patch, [227, 227])
    img_patch = img_patch.reshape((3, 227, 227))
    label = np.zeros((feat_bbox.shape[0],))
    return mx.io.NDArrayIter({'image_patch': [img_patch], 'feat_bbox': [feat_bbox]}, {'label': [label]},
                             batch_size=1, data_name=('image_patch', 'feat_bbox'), label_name=('label',))


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
        #print loss
        label = label.asnumpy()
        pred = pred.asnumpy()
        true_num = np.sum(pred.argmax(1) == label)
        false_num = label.shape[0] - true_num
        #print 'success:%d,fail:%d' % (true_num, false_num)

        self.sum_metric += true_num
        self.num_inst += label.shape[0]


def parse_args():
    parser = argparse.ArgumentParser(description='Train MDNet network')
    parser.add_argument('--gpu', help='GPU device to train with', default=1, type=int)
    parser.add_argument('--num_epoch', help='epoch of training', default=1, type=int)
    parser.add_argument('--lr', help='base learning rate', default=0.0001, type=float)
    parser.add_argument('--wd', help='base learning rate', default=0.005, type=float)
    parser.add_argument('--OTB_path', help='OTB folder', default='/Users/chenjunjie/workspace/OTB', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    seq_name = 'Surfer'
    otb = OTBHelper(args.OTB_path)
    img_list = otb.get_img(seq_name)[0:30]
    gt_list = otb.get_gt(seq_name)[0:30]
    ctx = mx.gpu(args.gpu)
    model = None
    begin_epoch = 0
    count = 0
    for img_path, gt in zip(img_list, gt_list):
        # validate
        val_img = cv2.imread(img_list[count + 1])
        val_gt = gt_list[count + 1]
        val_iter = get_train_iter(val_img, val_gt)

        model = train_on_one_frame(args, img_path, gt, model, begin_epoch, args.num_epoch, ctx, val_iter)
        begin_epoch += args.num_epoch
        count += 1
        # pred_region = test(model, '0002.jpg', [275, 137, 23, 26])
        # print pred_region


if __name__ == '__main__':
    main()
