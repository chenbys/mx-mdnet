# -*-coding:utf- 8-*-

import Queue
import mxnet as mx
import argparse
import numpy as np
import datahelper
import util
import extend
import random
from setting import const

update_data_queue = Queue.Queue(maxsize=100)


def get_update_data(frame_len, batch_num):
    '''
        返回最近frame_len帧 组成的 update_data
    :param frame_len: 长期100，短期20
    :return:
    '''
    total = update_data_queue.qsize()
    img_patches, feat_bboxes, labels = [], [], []

    for i in range(1, min(total, 3)):
        a, b, c = update_data_queue.queue[-i]
        img_patches += a
        feat_bboxes += b
        labels += c

    sel_idx = random.sample(range(0, const.update_batch_num), batch_num)
    for i in range(3, frame_len):
        a, b, c = update_data_queue.queue[-(i % total)]
        for idx in sel_idx:
            img_patches.append(a[idx])
            feat_bboxes.append(b[idx])
            labels.append(c[idx])
    return img_patches, feat_bboxes, labels


def add_update_data(img, gt, regions=[]):
    '''
        原版mdnet每一帧采50 pos 200 neg
        返回该帧构造出的 4 个img_patch, each 16 pos 32 neg
    :param img_patch:
    :param gt:
    :return:
    '''
    update_data = datahelper.get_update_data(img, gt, regions)
    if update_data_queue.full():
        update_data_queue.get()
    if update_data_queue.empty():
        update_data_queue.put(update_data)
    update_data_queue.put(update_data)


def offline_update(args, model, img, gt):
    data_batches = datahelper.get_data_batches(datahelper.get_train_data(img, gt))
    for epoch in range(0, args.num_epoch_for_offline):
        model = extend.train_with_hnm(model, data_batches, sel_factor=2)

    return model


def online_update(args, model, data_len, batch_num):
    data_batches = datahelper.get_data_batches(get_update_data(data_len, batch_num))
    for epoch in range(0, args.num_epoch_for_online):
        extend.train_with_hnm(model, data_batches, sel_factor=5)
    return model


def multi_track(model, img, pre_regions, topK=3):
    B, P = [], []

    single_track_topK = 2
    for pr in pre_regions:
        bboxes, probs = track(model, img, pr, topK=single_track_topK)
        B += bboxes
        P += probs

    B = np.array(B)
    P = np.array(P)
    top_idx = P.argsort()[-topK::]

    return B[top_idx].tolist(), P[top_idx].tolist()


def track(model, img, pre_region, topK=2):
    pred_data, restore_info = datahelper.get_predict_data(img, pre_region)

    pred_iter = datahelper.get_iter(pred_data)
    [img_patch], [feat_bboxes], [labels] = pred_data

    res = model.predict(pred_iter).asnumpy()
    pos_score = res[:, 1]
    top_idx = pos_score.argsort()[-topK::]

    patch_bboxes = util.feat2img(feat_bboxes[top_idx, 1:])
    img_bboxes = util.restore_bboxes(patch_bboxes, restore_info)

    return img_bboxes.tolist(), pos_score[top_idx].tolist()


def parse_args():
    parser = argparse.ArgumentParser(description='Train MDNet network')
    parser.add_argument('--gpu', help='GPU device to train with', default=0, type=int)
    parser.add_argument('--num_epoch_for_offline', default=2, type=int)
    parser.add_argument('--num_epoch_for_online', default=1, type=int)

    parser.add_argument('--fixed_conv', default=3, help='these params of [ conv_i <= ? ] will be fixed', type=int)
    parser.add_argument('--saved_fname', default='params/larger_wd_18000/shared', type=str)
    parser.add_argument('--OTB_path', help='OTB folder', default='/media/chen/datasets/OTB', type=str)
    parser.add_argument('--VOT_path', help='VOT folder', default='/home/chen/vot-toolkit/cmdnet-workspace/sequences',
                        type=str)
    parser.add_argument('--ROOT_path', help='cmd folder', default='/home/chen/mx-mdnet', type=str)

    parser.add_argument('--wd', default=1.5e0, help='weight decay', type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--lr_offline', default=1e-5, help='base learning rate', type=float)
    parser.add_argument('--lr_online', default=5e-5, help='base learning rate', type=float)

    args = parser.parse_args()
    return args
