# -*-coding:utf- 8-*-

import random
from time import time
import numpy as np
import util
from setting import const


def get_neg_feat_bboxes(ideal_feat_bbox=const.pred_ideal_feat_bbox, feat_size=const.pred_feat_size):
    '''
    :param ideal_feat_bbox:
    :param feat_size:
    :return:
    '''
    # return: bbox on feature map, in format of (0,x1,y1,x2,y2)
    l_x1, l_y1, l_x2, l_y2 = ideal_feat_bbox
    w, h = l_x2 - l_x1, l_y2 - l_y1

    feat_w, feat_h = feat_size
    stride_x1, stride_y1, stride_x2, stride_y2 = [w / 2., h / 2., w, h]
    feat_bboxes = []
    for x1 in np.arange(0, feat_w - w, stride_x1):
        for y1 in np.arange(0, feat_h - h, stride_y1):
            feat_bboxes.append([0, x1, y1, x1 + w, y1 + h])

    # 也就是说，dx1至少要移动w/2,至多移动2w
    DX1 = w + h
    stride_x1, stride_y1, stride_x2, stride_y2 = [w / 4., h / 4., w / 4., h / 4.]
    for dx1 in np.arange(max(-l_x1, -DX1), min(feat_w - l_x1, DX1 + 1), stride_x1):
        if abs(dx1) < w / 2:
            continue
        DY1 = DX1 - abs(dx1)
        for dy1 in np.arange(max(-l_y1, -DY1), min(feat_h - l_y1, DY1 + 1), stride_y1):
            if abs(dy1) < h / 2:
                continue
            x1, y1 = l_x1 + dx1, l_y1 + dy1
            feat_bboxes.append([0, x1, y1, x1 + w, y1 + h])

    return feat_bboxes


def get_pos_feat_bboxes(ideal_feat_bbox=const.pred_ideal_feat_bbox,
                        feat_size=const.pred_feat_size):
    l_x1, l_y1, l_x2, l_y2 = ideal_feat_bbox
    feat_w, feat_h = feat_size
    w, h = l_x2 - l_x1, l_y2 - l_y1
    feat_boxes = []

    DX1 = (w + h) / 4.
    stride_x1, stride_y1, stride_x2, stride_y2 = [w / 4., h / 4., w / 4., h / 4.]
    for dx1 in np.arange(max(-l_x1, -DX1), min(feat_w - l_x1, DX1 + 1), stride_x1):
        DY1 = DX1 - abs(dx1)
        for dy1 in np.arange(max(-l_y1, -DY1), min(feat_h - l_y1, DY1 + 1), stride_y1):
            DX2 = DY1 - abs(dy1)
            x1, y1 = l_x1 + dx1, l_y1 + dy1
            for dx2 in np.arange(max(x1 - l_x2 + 1, -DX2), min(feat_w - l_x2, DX2 + 1), stride_x2):
                DY2 = DX2 - dx2
                for dy2 in np.arange(max(y1 - l_y2 + 1, -DY2), min(feat_h - l_y2, DY2 + 1), stride_y2):
                    feat_boxes.append([0, x1, y1, l_x2 + dx2, l_y2 + dy2])
    return feat_boxes


def get_predict_feat_bboxes(ideal_feat_bbox=const.pred_ideal_feat_bbox,
                            feat_size=const.pred_feat_size):
    # return: bbox on feature map, in format of (0,x1,y1,x2,y2)

    l_x1, l_y1, l_x2, l_y2 = ideal_feat_bbox
    feat_w, feat_h = feat_size

    feat_boxes = list()

    DX1 = 13
    stride_x1, stride_y1, stride_x2, stride_y2 = [4, 4, 4, 4]
    for dx1 in np.arange(max(-l_x1, -DX1), min(feat_w - l_x1, DX1 + 1), stride_x1):
        DY1 = DX1 - abs(dx1)
        for dy1 in np.arange(max(-l_y1, -DY1), min(feat_h - l_y1, DY1 + 1), stride_y1):
            DX2 = DY1 - abs(dy1)
            x1, y1 = l_x1 + dx1, l_y1 + dy1
            for dx2 in np.arange(max(x1 - l_x2 + 1, -DX2), min(feat_w - l_x2, DX2 + 1), stride_x2):
                DY2 = DX2 - dx2
                for dy2 in np.arange(max(y1 - l_y2 + 1, -DY2), min(feat_h - l_y2, DY2 + 1), stride_y2):
                    feat_boxes.append([0, x1, y1, l_x2 + dx2, l_y2 + dy2])

    DX1 = 3
    stride_x1, stride_y1, stride_x2, stride_y2 = [2, 2, 2, 2]
    for dx1 in np.arange(max(-l_x1, -DX1), min(feat_w - l_x1, DX1 + 1), stride_x1):
        DY1 = DX1 - abs(dx1)
        for dy1 in np.arange(max(-l_y1, -DY1), min(feat_h - l_y1, DY1 + 1), stride_y1):
            DX2 = DY1 - abs(dy1)
            x1, y1 = l_x1 + dx1, l_y1 + dy1
            for dx2 in np.arange(max(x1 - l_x2 + 1, -DX2), min(feat_w - l_x2, DX2 + 1), stride_x2):
                DY2 = DX2 - dx2
                for dy2 in np.arange(max(y1 - l_y2 + 1, -DY2), min(feat_h - l_y2, DY2 + 1), stride_y2):
                    feat_boxes.append([0, x1, y1, l_x2 + dx2, l_y2 + dy2])
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
