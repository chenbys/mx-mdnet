import numpy as np
import util


def sample_on_feat(stride_x=2, stride_y=2, stride_w=2, stride_h=2,
                   ideal_w=12, ideal_h=12, feat_w=24, feat_h=24):
    '''

    :param stride_x:
    :param stride_y:
    :param stride_w:
    :param stride_h:
    :param feat_w:
    :param feat_h:
    :return: bbox on feature map, in format of (0,x1,y1,x2,y2)
    '''
    feat_boxes = list()
    for x in np.arange(0, feat_w - ideal_w / 2., stride_x):
        for y in np.arange(0, feat_h - ideal_h / 2., stride_y):
            max_w = min(ideal_w * 1.5, feat_w - x)
            max_h = min(ideal_h * 1.5, feat_h - y)
            for w in np.arange(ideal_w * 0.5, max_w + 0.1, stride_w):
                for h in np.arange(ideal_h * 0.5, max_h + 0.1, stride_h):
                    feat_boxes.append([0, x, y, x + w - 1, y + h - 1])

    return np.array(feat_boxes)


def get_samples(label_feat, pos_number=200, neg_number=200):
    import random

    x, y, w, h = label_feat[0, :]
    feat_bboxes = sample_on_feat(1, 1, 1, 1, w, h)
    feat_bboxes_ = util.x1y2x2y22xywh(feat_bboxes[:, 1:5])
    rat = util.overlap_ratio(label_feat, feat_bboxes_)
    pos_samples = feat_bboxes[rat > 0.7, :]
    neg_samples = feat_bboxes[rat < 0.3, :]
    # print 'pos:%d ,neg:%d, all:%d;' % (pos_samples.shape[0], neg_samples.shape[0], feat_bboxes.shape[0])
    # select samples
    # ISSUE: what if pos_samples.shape[0] < pos_number?
    pos_select_index = random.sample(range(0, pos_samples.shape[0]), pos_number)
    neg_select_index = random.sample(range(0, neg_samples.shape[0]), pos_number)

    return np.vstack((pos_samples[pos_select_index], neg_samples[neg_select_index])), \
           np.hstack((np.ones((pos_number,)), np.zeros((neg_number,))))


def get_samples_with_iou_label(label_feat, pos_number=4, neg_number=4):
    import mxnet as mx
    x, y, w, h = label_feat[0, :]
    feat_bboxes = sample_on_feat(1, 1, 1, 1, w, h)
    feat_bboxes_ = util.x1y2x2y22xywh(feat_bboxes[:, 1:5])
    rat = util.overlap_ratio(label_feat, feat_bboxes_)
    pos_idx = mx.ndarray.topk(mx.ndarray.array(rat), axis=0, k=pos_number).asnumpy().astype('int32')
    neg_idx = mx.ndarray.topk(mx.ndarray.array(rat * -1), axis=0, k=neg_number).asnumpy().astype('int32')
    all_idx=np.hstack((pos_idx, neg_idx))

    return feat_bboxes[all_idx,:], rat[all_idx]
