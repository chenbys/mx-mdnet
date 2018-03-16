# -*-coding:utf- 8-*-

import random

import numpy as np
import util
import matplotlib.pyplot as plt


def get_train_feat_sample(stride_x=2, stride_y=2, stride_w=2, stride_h=2,
                          ideal_x=6, ideal_y=6, ideal_w=9, ideal_h=9,
                          feat_w=20, feat_h=20):
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

    min_x = max(0, round(ideal_x - ideal_w / 2.))
    max_x = min(feat_w, round(ideal_x + ideal_w / 2.))
    min_y = max(0, ideal_y - ideal_h / 2.)
    max_y = min(feat_h, round(ideal_y + ideal_h / 2.))

    for x in np.arange(min_x, max_x, stride_x):
        for y in np.arange(min_y, max_y, stride_y):

            min_w = round(0.4 * ideal_w)
            max_w = min(ideal_w * 2, feat_w - x)
            min_h = round(0.4 * ideal_h)
            max_h = min(ideal_h * 2, feat_h - y)

            for w in np.arange(min_w, max_w, stride_w):
                for h in np.arange(min_h, max_h, stride_h):
                    feat_boxes.append([0, x, y, x + w - 1, y + h - 1])

    return np.array(feat_boxes)


def get_predict_feat_sample():
    # return: bbox on feature map, in format of (0,x1,y1,x2,y2)

    feat_boxes = list()
    # 目标小变化的候选区域
    for x1 in range(4, 9, 2):
        for y1 in range(4, 9, 2):
            for x2 in range(12, 17, 2):
                for y2 in range(12, 17, 2):
                    feat_boxes.append([0, x1, y1, x2, y2])
    # 目标大变化的候选区域
    for x1 in range(0, 20, 3):
        for y1 in range(0, 20, 3):
            for x2 in range(x1 + 5, min(x1 + 15, 20), 3):
                for y2 in range(y1 + 5, min(y1 + 15, 20), 3):
                    feat_boxes.append([0, x1, y1, x2, y2])
    return np.array(feat_boxes)


def get_01samples(patch_gt, pos_number=100, neg_number=500):
    '''
        Q: patch上的重叠率作为label，和还原到原图上的重叠率做label一样吗
    :param patch_gt:
    :param pos_number:
    :param neg_number:
    :return:
    '''
    label_feat = util.x1y2x2y22xywh(util.img2feat(util.xywh2x1y1x2y2(patch_gt)))
    x, y, w, h = label_feat[0, :]
    # pos
    pos_bboxes = get_train_feat_sample(1, 1, 1, 1, x, y, w, h)
    pos_patch_bboxes = util.feat2img(pos_bboxes[:, 1:])
    rat = util.overlap_ratio(patch_gt, pos_patch_bboxes)
    pos_samples = pos_bboxes[rat > 0.65, :]
    pos_select_index = rand_sample(np.arange(0, pos_samples.shape[0]), pos_number)

    # neg
    neg_bboxes = get_train_feat_sample(2, 2, 2, 2)
    neg_patch_bboxes = util.feat2img(neg_bboxes[:, 1:])
    rat = util.overlap_ratio(patch_gt, neg_patch_bboxes)
    neg_samples = neg_bboxes[rat < 0.5, :]
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
