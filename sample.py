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


def get_samples_with_iou_label(label_feat, p_number=500, h_number=300, m_number=200, s_number=200):
    '''
    :param label_feat:
    :param pos_number:
    :param rand_number:
    :param neg_number:
    :return:
    '''
    import mxnet as mx
    import random

    x, y, w, h = label_feat[0, :]
    # generate pos samples
    feat_bboxes = sample_on_feat(1, 1, 1, 1, w, h)
    feat_bboxes_ = util.x1y2x2y22xywh(feat_bboxes[:, 1:5])
    rat = util.overlap_ratio(label_feat, feat_bboxes_)
    # for p
    p_samples, p_labels = feat_bboxes[rat > 0.8, :], rat[rat > 0.8]
    num = p_samples.shape[0]
    A, B = p_number / num, p_number % num
    p_idx = random.sample(range(0, num), B)
    p_samples = np.vstack((np.repeat(p_samples, A, axis=0), p_samples[p_idx, :]))
    p_labels = np.hstack((np.repeat(p_labels, A, axis=0), p_labels[p_idx]))
    # for h
    h_samples, h_labels = feat_bboxes[rat > 0.7, :], rat[rat > 0.7]
    num = h_samples.shape[0]
    A, B = h_number / num, h_number % num
    h_idx = random.sample(range(0, num), B)
    h_samples = np.vstack((np.repeat(h_samples, A, axis=0), h_samples[h_idx, :]))
    h_labels = np.hstack((np.repeat(h_labels, A, axis=0), h_labels[h_idx]))
    # for m
    m_samples, m_labels = feat_bboxes[0.7 > rat, :], rat[0.7 > rat]
    num = m_samples.shape[0]
    A, B = m_number / num, m_number % num
    m_idx = random.sample(range(0, num), B)
    m_samples = np.vstack((np.repeat(m_samples, A, axis=0), m_samples[m_idx, :]))
    m_labels = np.hstack((np.repeat(m_labels, A, axis=0), m_labels[m_idx]))
    # for s
    s_samples, s_labels = feat_bboxes[0.3 > rat, :], rat[0.3 > rat]
    num = s_samples.shape[0]
    A, B = s_number / num, s_number % num
    s_idx = random.sample(range(0, num), B)
    s_samples = np.vstack((np.repeat(s_samples, A, axis=0), s_samples[s_idx, :]))
    s_labels = np.hstack((np.repeat(s_labels, A, axis=0), s_labels[s_idx]))

    return np.vstack((p_samples, h_samples, m_samples, s_samples)), \
           np.hstack((p_labels, h_labels, m_labels, s_labels))
