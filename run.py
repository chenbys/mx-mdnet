# -*-coding:utf- 8-*-

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


def debug_track_seq(args, model, img_paths, gts):
    # train on first frame
    for j in range(3):
        for i in range(args.num_frame_for_offline):
            print 'train offine on frame %d' % i
            train_img_path, train_gt = img_paths[i], gts[i]
            eval_img_path, eval_gt = img_paths[i + 2], gts[i + 2]
            t = time.time()
            train_iter = datahelper.get_train_iter(datahelper.get_train_data(train_img_path, train_gt))
            logging.getLogger().info('time cost for getting one train iter :%f' % (time.time() - t))
            eval_iter = datahelper.get_train_iter(
                datahelper.get_train_data(eval_img_path, eval_gt))

            model.fit(train_data=train_iter, eval_data=eval_iter,
                      optimizer='sgd',
                      eval_metric=mx.metric.CompositeEvalMetric(
                          [extend.PR(0.7), extend.RR(0.7), extend.TrackTopKACC(10, 0.6)]),
                      optimizer_params={'learning_rate': args.lr_offline,
                                        'wd': args.wd,
                                        'momentum': args.momentum,
                                        # 'clip_gradient': 5,
                                        'lr_scheduler': extend.MDScheduler(
                                            args.lr_step, args.lr_factor, args.lr_stop)},
                      begin_epoch=j * 30, num_epoch=j * 30 + args.num_epoch_for_offline)
            track(model, img_paths[0], pre_region=gts[0], gt=gts[0])
            track(model, img_paths[1], pre_region=gts[1], gt=gts[1])
            track(model, img_paths[2], pre_region=gts[2], gt=gts[2])
            track(model, img_paths[3], pre_region=gts[3], gt=gts[3])
            track(model, img_paths[5], pre_region=gts[5], gt=gts[5])

            # [256.0, 152.0, 73.0, 210.0] for Liquor
    # config.gt = [262, 94, 16, 26]
    # track(model, '/media/chen/datasets/OTB/Biker/img/0001.jpg', [262, 94, 16, 26])
    # config.gt = [256.0, 152.0, 73.0, 210.0]
    # track(model, img_paths[2], [256.0, 152.0, 100.0, 210.0])
    a = model.score(datahelper.get_val_iter(
        datahelper.get_val_data(img_paths[0], pre_region=gts[0], gt=gts[0])),
        mx.metric.CompositeEvalMetric([extend.PR(0.7), extend.RR(0.7), extend.TrackTopKACC(10, 0.6)]))

    res = []
    scores = []
    length = len(img_paths)
    region = gts[0]

    for cur in range(1, length):
        T = time.time()
        # track
        region, score = track(model, img_paths[cur], pre_region=region, gt=gts[cur])

        res.append(region)

        # report
        logging.getLogger().info(
            '@CHEN-> IOU : -> %.2f <- !!!  score: %.2f for tracking on frame %d, cost %4.2f' \
            % (util.overlap_ratio(gts[cur], region), score, cur, time.time() - T))

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

        # online update
        scores.append(score)
        if score < 0.9:
            # short term update
            logging.getLogger().info('@CHEN->short term update')
            model = online_update(args, model, img_paths, res, cur,
                                  num_epoch=args.num_epoch_for_online)
        elif cur % 10 == 0:
            # long term update
            logging.getLogger().info('@CHEN->long term update')
            model = online_update(args, model, img_paths, res, cur,
                                  num_epoch=args.num_epoch_for_online)

    return res, scores


def track_seq(args, model, img_paths, first_gt):
    # train on first frame
    model = train_on_first(args, model, img_paths[0], first_gt,
                           num_epoch=args.num_epoch_for_offline)

    res = [first_gt]
    scores = [0]
    length = len(img_paths)
    region = first_gt
    for cur in range(length):
        img_path = img_paths[cur]

        # track
        region, score = track(model, img_path, region, topk=5)
        res.append(region)

        # online update
        scores.append(score)
        if score < 0.5:
            # short term update
            print '@CHEN->short term update'
            model = online_update(args, model, img_paths, res, cur,
                                  num_epoch=args.num_epoch_for_online)
        elif cur % 10 == 0:
            # long term update
            print '@CHEN->long term update'
            model = online_update(args, model, img_paths, res, cur,
                                  num_epoch=args.num_epoch_for_online)

    return res, scores


def online_update(args, model, img_paths, res, cur, history_len=10, num_epoch=10):
    return model
    for i in range(max(0, cur - history_len), cur + 1):
        metric = mx.metric.CompositeEvalMetric()
        metric.add(extend.MDNetIOUACC())
        metric.add(extend.MDNetIOULoss())
        train_iter = datahelper.get_train_iter(datahelper.get_train_data(img_paths[i], res[i]))
        model.fit(train_data=train_iter, optimizer='sgd',
                  optimizer_params={'learning_rate': args.lr_online,
                                    'wd': args.wd,
                                    'momentum': args.momentum,
                                    # 'clip_gradient': 5,
                                    'lr_scheduler': extend.MDScheduler(
                                        args.lr_step, args.lr_factor, args.lr_stop)},
                  eval_metric=metric, begin_epoch=0, num_epoch=num_epoch)
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

    show_tracking()
    return opt_img_bbox, opt_score


def debug_track_on_OTB():
    args = parse_args()
    config.ctx = mx.gpu(args.gpu)

    otb = datahelper.OTBHelper(args.OTB_path)
    img_paths, gts = otb.get_seq('Girl')

    # for debug and check
    config.gts = gts
    config.img_paths = img_paths
    datahelper.get_predict_data(img_paths[0], gts[0])

    model, all_params = extend.init_model(args)

    logging.getLogger().setLevel(logging.DEBUG)
    res, scores = debug_track_seq(args, model, img_paths, gts)


def parse_args():
    parser = argparse.ArgumentParser(description='Train MDNet network')
    parser.add_argument('--gpu', help='GPU device to train with', default=0, type=int)
    parser.add_argument('--num_epoch_for_offline', default=30, type=int)
    parser.add_argument('--num_epoch_for_online', default=0, help='epoch of training for every frame', type=int)
    parser.add_argument('--num_frame_for_offline', default=1, help='epoch of training for every frame', type=int)
    parser.add_argument('--lr_online', help='base learning rate', default=1e-5, type=float)
    parser.add_argument('--wd', default=1e1, help='weight decay', type=float)
    parser.add_argument('--OTB_path', help='OTB folder', default='/media/chen/datasets/OTB', type=str)
    parser.add_argument('--VOT_path', help='VOT folder', default='/media/chen/datasets/VOT2015', type=str)
    parser.add_argument('--lr_step', default=222 * 15, help='every 121 num for one epoch', type=int)
    parser.add_argument('--lr_factor', default=0.5, help='20 times will be around 0.1', type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--lr_stop', default=5e-8, type=float)
    parser.add_argument('--lr_offline', default=2e-7, help='base learning rate', type=float)
    parser.add_argument('--fixed_conv', help='the params before(include) which conv are all fixed',
                        default=3, type=int)
    parser.add_argument('--saved_fname', default='conv123fc4fc5', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    debug_track_on_OTB()
