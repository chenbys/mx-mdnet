# -*-coding:utf- 8-*-
import Queue

import mxnet as mx
import argparse
import numpy as np
import copy

import time

import eva
import sample
import datahelper
import util
import extend
from setting import config
import matplotlib.pyplot as plt
from matplotlib import patches
import logging

update_data_queue = Queue.Queue(maxsize=100)


def debug_track_seq(args, model, img_paths, gts):
    '''

    :param args:
    :param model:
    :param img_paths: 待跟踪的图片地址list，首个是有标注的，用来首帧训练的。
    :param gts: 用来调试的每帧的gt，本应只传gts[0]
    :return:
    '''
    # train on first frame
    print 'train offine on frame 0'
    train_img_path, train_gt = img_paths[0], gts[0]
    eval_img_path, eval_gt = img_paths[5], gts[5]
    t = time.time()
    train_iter = datahelper.get_train_iter(datahelper.get_train_data(train_img_path, train_gt))
    print('time cost for getting one train iter :%f' % (time.time() - t))

    eval_iter = datahelper.get_train_iter(
        datahelper.get_train_data(eval_img_path, eval_gt))

    model.fit(train_data=train_iter, eval_data=eval_iter,
              optimizer='sgd',
              eval_metric=mx.metric.CompositeEvalMetric(
                  [extend.PR(0.5), extend.RR(0.5), extend.TrackTopKACC(10, 0.6)]),
              optimizer_params={'learning_rate': args.lr_offline,
                                'wd': args.wd,
                                'momentum': args.momentum,
                                # 'clip_gradient': 5,
                                },
              begin_epoch=0, num_epoch=args.num_epoch_for_offline)

    # res, scores 是保存每一帧的结果位置和给出的是目标的概率的list，包括用来训练的首帧
    res, probs = [gts[0]], [1]
    region = gts[0]

    # prepare online update data
    add_update_data(img_paths[0], gts[0])

    length = len(img_paths)
    for cur in range(1, length):
        T = time.time()
        # track
        region, prob = track(model, img_paths[cur], pre_region=region, gt=gts[cur])

        res.append(region)
        probs.append(prob)

        # report
        logging.getLogger().info(
            '@CHEN-> IOU : [ %.2f ] !!!  prob: %.2f for tracking on frame %d, cost %4.4f' \
            % (util.overlap_ratio(gts[cur], region), prob, cur, time.time() - T))

        # show
        def show_tracking():
            img_path = img_paths[cur]
            gt = gts[cur]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(plt.imread(img_path))
            ax.add_patch(patches.Rectangle((region[0], region[1]), region[2], region[3],
                                           linewidth=4, edgecolor='red', facecolor='none'))
            ax.add_patch(patches.Rectangle((gt[0], gt[1]), gt[2], gt[3],
                                           linewidth=1, edgecolor='blue', facecolor='none'))
            fig.show()

        # prepare online update data
        add_update_data(img_paths[cur], res[cur])

        # online update
        if prob < 0.5:
            # short term update
            logging.getLogger().info('@CHEN->short term update')
            model = online_update(args, model, 20)
        elif cur % 10 == 0:
            # long term update
            logging.getLogger().info('@CHEN->long term update')
            model = online_update(args, model, 100)

    return res, probs


def get_update_data(frame_len=20):
    '''
        返回最近frame_len帧 组成的 update_data
    :param frame_len: 长期100，短期20
    :return:
    '''
    frame_len = min(frame_len, update_data_queue.qsize())
    img_patches, feat_bboxes, labels = [], [], []
    for i in range(1, frame_len + 1):
        a, b, c = update_data_queue.queue[-i]
        img_patches += a
        feat_bboxes += b
        labels += c
    return img_patches, feat_bboxes, labels


def add_update_data(img_patch, gt):
    '''
        原版mdnet每一帧采50 pos 200 neg
        返回该帧构造出的 4 个img_patch, each 16 pos 32 neg
    :param img_patch:
    :param gt:
    :return:
    '''
    if update_data_queue.full():
        update_data_queue.get()

    update_data = datahelper.get_update_data(img_patch, gt)
    update_data_queue.put(update_data)


def online_update(args, model, data_len=20):
    '''
        pos sample 只用短期的，因为老旧的负样本是无关的。（如果速度允许的话，为了省事，都更新应该影响不大吧。）
        mdnet：long term len 100F, short term len 20F（感觉短期有点太长了吧，可能大多变化都在几帧之内完成）

        用一个list保存每一帧对应的update_data, 每一帧有几个 batch，每个batch 几个img_patch，每个img_patch 32 pos 32 neg

        long term: every 10 frames
            利用近长期帧组成 batch, each 32 pos, 96 neg
        short term: score < 0
            利用近短期帧组成 batch, each 32 pos, 96 neg

    :param args:
    :param model:
    :param img_paths:
    :param res:
    :param cur:
    :param history_len:
    :param num_epoch:
    :return:
    '''
    update_iter = datahelper.get_train_iter(get_update_data(data_len))
    model.fit(train_data=update_iter, optimizer='sgd',
              eval_metric=mx.metric.CompositeEvalMetric(
                  [extend.PR(0.5), extend.RR(0.5), extend.TrackTopKACC(10, 0.6)]),
              optimizer_params={'learning_rate': args.lr_offline,
                                'wd': args.wd,
                                'momentum': args.momentum,
                                # 'clip_gradient': 5,
                                },
              begin_epoch=0, num_epoch=args.num_epoch_for_online)
    return model


def track(model, img_path, pre_region, gt):
    pred_data, restore_info = datahelper.get_predict_data(img_path, pre_region)
    pred_iter = datahelper.get_predict_iter(pred_data)
    img_patch, feat_bboxes, labels = pred_data
    res = model.predict(pred_iter).asnumpy()
    pos_score = res[:, 1]

    patch_bboxes = util.feat2img(feat_bboxes[:, 1:])
    img_bboxes = util.restore_img_bbox(patch_bboxes, restore_info)
    labels = util.overlap_ratio(gt, img_bboxes)

    if 1:
        # 按照输出概率的最大topK个的bbox来平均出结果
        topK = 5
        top_idx = pos_score.argsort()[-topK::]
    else:
        # 按照输出概率大于0.9的所有bbox来平均出结果
        top_idx = pos_score > 0.9

    top_scores = pos_score[top_idx]
    top_feat_bboxes = feat_bboxes[top_idx, 1:]
    top_patch_bboxes = util.feat2img(top_feat_bboxes)

    def check_pred_data(i):
        feat_bbox = feat_bboxes[i, 1:].reshape(1, 4)
        patch_bbox = util.feat2img(feat_bbox)
        img_bbox = util.restore_img_bbox(patch_bbox, restore_info).reshape(4, )

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(plt.imread(img_path))
        ax.add_patch(patches.Rectangle((img_bbox[0], img_bbox[1]), img_bbox[2], img_bbox[3],
                                       linewidth=4, edgecolor='red', facecolor='none'))
        ax.add_patch(patches.Rectangle((gt[0], gt[1]), gt[2], gt[3],
                                       linewidth=1, edgecolor='blue', facecolor='none'))
        fig.show()
        return (pos_score[i], labels[i])

    # [201 215 270 198 202]
    top_img_bboxes = util.restore_img_bbox(top_patch_bboxes, restore_info)
    opt_img_bbox = np.mean(top_img_bboxes, 0)
    opt_score = top_scores.mean()

    def plot():
        plt.plot(pos_score, 'r')
        plt.plot(labels, 'blue')

    def show_tracking():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(plt.imread(img_path))
        ax.add_patch(patches.Rectangle((opt_img_bbox[0], opt_img_bbox[1]), opt_img_bbox[2], opt_img_bbox[3],
                                       linewidth=4, edgecolor='red', facecolor='none'))
        ax.add_patch(patches.Rectangle((gt[0], gt[1]), gt[2], gt[3],
                                       linewidth=1, edgecolor='blue', facecolor='none'))
        ax.add_patch(patches.Rectangle((pre_region[0], pre_region[1]), pre_region[2], pre_region[3],
                                       linewidth=1, edgecolor='yellow', facecolor='none'))
        fig.show()

    # show_tracking()
    return opt_img_bbox, opt_score


def debug_track_on_OTB():
    args = parse_args()
    config.ctx = mx.gpu(args.gpu)

    otb = datahelper.OTBHelper(args.OTB_path)
    img_paths, gts = otb.get_seq('Diving')

    # for debug and check
    config.gts = gts
    config.img_paths = img_paths

    # debug
    # Diving 0042.jpg 需要short term update
    model, all_params = extend.init_model(args)

    logging.getLogger().setLevel(logging.DEBUG)
    res, scores = debug_track_seq(args, model, img_paths, gts)


def parse_args():
    parser = argparse.ArgumentParser(description='Train MDNet network')
    parser.add_argument('--gpu', help='GPU device to train with', default=0, type=int)
    parser.add_argument('--num_epoch_for_offline', default=10, type=int)
    parser.add_argument('--num_epoch_for_online', default=3, type=int)
    parser.add_argument('--fixed_conv', help='these params of [ conv_i <= ? ] will be fixed', default=3, type=int)
    parser.add_argument('--saved_fname', default='conv123fc4fc5', type=str)

    parser.add_argument('--OTB_path', help='OTB folder', default='/media/chen/datasets/OTB', type=str)
    parser.add_argument('--VOT_path', help='VOT folder', default='/media/chen/datasets/VOT2015', type=str)
    parser.add_argument('--lr_step', default=222 * 15, help='every 121 num for one epoch', type=int)
    parser.add_argument('--lr_factor', default=0.5, help='20 times will be around 0.1', type=float)
    parser.add_argument('--lr_stop', default=5e-8, type=float)

    parser.add_argument('--wd', default=1e-3, help='weight decay', type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--lr_offline', default=2e-5, help='base learning rate', type=float)
    parser.add_argument('--lr_online', default=1e-5, help='base learning rate', type=float)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    debug_track_on_OTB()
