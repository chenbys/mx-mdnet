# -*-coding:utf- 8-*-

import random
from time import time
import numpy as np
import util
from setting import const, config


def get_train_samples(patch_gt, pos_number=32, neg_number=96):
    '''
    :param patch_gt:
    :param pos_number:
    :param neg_number:
    :return:
    '''
    label_feat = util.img2feat(util.xywh2x1y1x2y2(patch_gt))[0, :]
    feat_bboxes = get_train_feat_bboxes(label_feat, strides=[1, 1, 1, 1])
    patch_bboxes = util.feat2img(feat_bboxes[:, 1:])
    rat = util.overlap_ratio(patch_gt, patch_bboxes)
    # pos
    pos_samples = feat_bboxes[rat > config.train_pos_th, :]
    pos_select_index = rand_sample(np.arange(0, pos_samples.shape[0]), pos_number)
    # neg
    neg_samples = feat_bboxes[rat < config.train_neg_th, :]
    neg_select_index = rand_sample(np.arange(0, neg_samples.shape[0]), neg_number)

    a, b = np.vstack((pos_samples[pos_select_index], neg_samples[neg_select_index])), \
           np.hstack((np.ones((pos_number,)), np.zeros((neg_number,))))
    return a, b


def get_train_feat_bboxes(labal_feat_bbox,
                          strides=[2, 2, 2, 2],
                          feat_size=const.feat_size):
    '''

    :param stride_x:
    :param stride_y:
    :param stride_w:
    :param stride_h:
    :param feat_w:
    :param feat_h:
    :return: bbox on feature map, in format of (0,x1,y1,x2,y2)
    '''
    stride_x1, stride_y1, stride_x2, stride_y2 = strides
    l_x1, l_y1, l_x2, l_y2 = labal_feat_bbox
    feat_w, feat_h = feat_size

    feat_boxes = list()

    DX1 = 10
    for dx1 in np.arange(max(-l_x1, -DX1), DX1 + 1, stride_x1):
        DY1 = DX1 - abs(dx1)
        for dy1 in np.arange(max(-l_y1, -DY1), DY1 + 1, stride_y1):
            DX2 = DY1 - abs(dy1)
            for dx2 in np.arange(-DX2, min(feat_w - l_x2, DX2 + 1), stride_x2):
                DY2 = DX2 - dx2
                for dy2 in np.arange(-DY2, min(feat_h - l_y2, DY2 + 1), stride_y2):
                    feat_boxes.append([0, l_x1 + dx1, l_y1 + dy1, l_x2 + dx2, l_y2 + dy2])

    return np.array(feat_boxes)


def get_update_feat_bboxes(labal_feat_bbox,
                           strides=[2, 2, 2, 2],
                           feat_size=const.feat_size):
    T = time()
    stride_x1, stride_y1, stride_x2, stride_y2 = strides
    l_x1, l_y1, l_x2, l_y2 = labal_feat_bbox
    feat_w, feat_h = feat_size

    feat_boxes = list()

    DX1 = 10
    for dx1 in np.arange(max(-l_x1, -DX1), DX1 + 1, stride_x1):
        DY1 = DX1 - abs(dx1)
        for dy1 in np.arange(max(-l_y1, -DY1), DY1 + 1, stride_y1):
            DX2 = DY1 - abs(dy1)
            for dx2 in np.arange(-DX2, min(feat_w - l_x2, DX2 + 1), stride_x2):
                DY2 = DX2 - dx2
                for dy2 in np.arange(-DY2, min(feat_h - l_y2, DY2 + 1), stride_y2):
                    feat_boxes.append([0, l_x1 + dx1, l_y1 + dy1, l_x2 + dx2, l_y2 + dy2])
    # print 'Time for get update feat bboxes:%.6f' % (time() - T)
    return np.array(feat_boxes)


def get_update_samples(patch_gt, pos_number=16, neg_number=32):
    label_feat = util.img2feat(util.xywh2x1y1x2y2(patch_gt))[0, :]
    feat_bboxes = get_update_feat_bboxes(label_feat, strides=[2, 2, 2, 2])

    patch_bboxes = util.feat2img(feat_bboxes[:, 1:])
    rat = util.overlap_ratio(patch_gt, patch_bboxes)
    # pos
    pos_samples = feat_bboxes[rat > config.update_pos_th, :]
    pos_select_index = rand_sample(np.arange(0, pos_samples.shape[0]), pos_number)
    # neg
    neg_samples = feat_bboxes[rat < config.update_neg_th, :]
    neg_select_index = rand_sample(np.arange(0, neg_samples.shape[0]), neg_number)

    a, b = np.vstack((pos_samples[pos_select_index], neg_samples[neg_select_index])), \
           np.hstack((np.ones((pos_number,)), np.zeros((neg_number,))))

    return a, b


def get_predict_feat_bboxes(strides=[2, 2, 2, 2], ideal_feat_bbox=const.ideal_feat_bbox, feat_size=const.feat_size):
    T = time()
    # return: bbox on feature map, in format of (0,x1,y1,x2,y2)

    stride_x1, stride_y1, stride_x2, stride_y2 = strides
    l_x1, l_y1, l_x2, l_y2 = ideal_feat_bbox
    feat_w, feat_h = feat_size

    feat_boxes = list()

    DX1 = 6
    for dx1 in np.arange(max(-l_x1, -DX1), DX1 + 1, stride_x1):
        DY1 = DX1 - abs(dx1)
        for dy1 in np.arange(max(-l_y1, -DY1), DY1 + 1, stride_y1):
            DX2 = DY1 - abs(dy1)
            for dx2 in np.arange(-DX2, min(feat_w - l_x2, DX2 + 1), stride_x2):
                DY2 = DX2 - dx2
                for dy2 in np.arange(-DY2, min(feat_h - l_y2, DY2 + 1), stride_y2):
                    feat_boxes.append([0, l_x1 + dx1, l_y1 + dy1, l_x2 + dx2, l_y2 + dy2])
    # print 'Time for get predict feat bboxes:%.6f' % (time() - T)
    return np.array(feat_boxes)


def rand_sample(pop, num):
    '''
        still work when sample num > pop

    :param pop:
    :param num:
    :return:
    '''
    pop_size = pop.shape[0]
    A, B = num / pop_size, num % pop_size

    sample_idx = random.sample(pop, B)

    if A == 0:
        return pop[sample_idx]
    else:
        # print 'not enough: %d, acquire: %d' % (pop_size, num)
        return np.hstack((np.repeat(pop, A, axis=0), pop[sample_idx]))
