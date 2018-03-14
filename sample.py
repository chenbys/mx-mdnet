# -*-coding:utf- 8-*-

import random

import numpy as np
import util
import matplotlib.pyplot as plt


def get_train_feat_sample(stride_x=2, stride_y=2, stride_w=2, stride_h=2,
                          ideal_w=9, ideal_h=9, feat_w=20, feat_h=20):
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
    for x in np.arange(0, feat_w - ideal_w / 3., stride_x):
        for y in np.arange(0, feat_h - ideal_h / 3., stride_y):
            max_w = min(ideal_w * 2, feat_w - x)
            max_h = min(ideal_h * 2, feat_h - y)
            for w in np.arange(ideal_w * 0.4, max_w + 0.1, stride_w):
                for h in np.arange(ideal_h * 0.4, max_h + 0.1, stride_h):
                    feat_boxes.append([0, int(x), int(y), int(x + w - 1), int(y + h - 1)])

    return np.array(feat_boxes)


def get_predict_feat_sample():
    # return: bbox on feature map, in format of (0,x1,y1,x2,y2)

    feat_boxes = list()
    # 目标小变化的候选区域
    for x1 in range(4, 8):
        for y1 in range(4, 8):
            for x2 in range(12, 16):
                for y2 in range(12, 16):
                    feat_boxes.append([0, x1, y1, x2, y2])
    # 目标大变化的候选区域
    for x1 in range(0, 20, 3):
        for y1 in range(0, 20, 3):
            for x2 in range(x1 + 5, min(x1 + 15, 20), 3):
                for y2 in range(y1 + 5, min(y1 + 15, 20), 3):
                    feat_boxes.append([0, x1, y1, x2, y2])
    return np.array(feat_boxes)


def get_01samples(patch_gt, pos_number=50, neg_number=500):
    '''
        Q: patch上的重叠率作为label，和还原到原图上的重叠率做label一样吗
    :param patch_gt:
    :param pos_number:
    :param neg_number:
    :return:
    '''
    # label_feat = util.x1y2x2y22xywh(util.img2feat(util.xywh2x1y1x2y2(patch_gt)))
    # x, y, w, h = label_feat[0, :]
    # generate pos samples
    feat_bboxes = get_train_feat_sample(1, 1, 1, 1)
    patch_bboxes = util.feat2img(feat_bboxes[:, 1:])
    rat = util.overlap_ratio(patch_gt, patch_bboxes)

    pos_samples = feat_bboxes[rat > 0.7, :]
    neg_samples = feat_bboxes[rat < 0.5, :]
    # print 'pos:%d ,neg:%d, all:%d;' % (pos_samples.shape[0], neg_samples.shape[0], feat_bboxes.shape[0])
    # select samples
    # ISSUE: what if pos_samples.shape[0] < pos_number?
    pos_select_index = rand_sample(np.arange(0, pos_samples.shape[0]), pos_number)
    neg_select_index = rand_sample(np.arange(0, neg_samples.shape[0]), neg_number)
    a, b = np.vstack((pos_samples[pos_select_index], neg_samples[neg_select_index])), \
           np.hstack((np.ones((pos_number,)), np.zeros((neg_number,))))
    return a, b


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
        print 'not enough: %d, acquire: %d' % (pop_size, num)
        return np.hstack((np.repeat(pop, A, axis=0), pop[sample_idx]))
