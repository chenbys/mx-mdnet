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
    for j in range(1):
        for i in range(args.num_frame_for_offline):
            print 'train offine on frame %d' % i
            train_img_path, train_gt = img_paths[i], gts[i]
            eval_img_path, eval_gt = img_paths[i + 1], gts[i + 1]

            train_iter = datahelper.get_train_iter(datahelper.get_train_data(train_img_path, train_gt))
            eval_iter = datahelper.get_train_iter(datahelper.get_train_data(eval_img_path, eval_gt))

            mdmetric = extend.MDNetACC()
            model.fit(train_data=train_iter, eval_data=eval_iter,
                      optimizer='sgd', eval_metric=mdmetric,
                      optimizer_params={'learning_rate': args.lr_offline,
                                        'wd': args.wd,
                                        'momentum': args.momentum,
                                        # 'clip_gradient': 5,
                                        'lr_scheduler': extend.MDScheduler(
                                            args.lr_step, args.lr_factor, args.lr_stop)},
                      begin_epoch=0, num_epoch=args.num_epoch_for_offline)

    res = []
    scores = []
    length = len(img_paths)
    region = gts[0]

    def check_tracking(cur, res_region):
        img_path = img_paths[cur]
        gt = gts[cur]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(plt.imread(img_path))
        ax.add_patch(patches.Rectangle((res_region[0], res_region[1]), res_region[2], res_region[3],
                                       linewidth=4, edgecolor='red', facecolor='none'))
        ax.add_patch(patches.Rectangle((gt[0], gt[1]), gt[2], gt[3],
                                       linewidth=1, edgecolor='blue', facecolor='none'))
        fig.show()

    for cur in range(1, length):
        T = time.time()
        # track
        region, score = track(model, img_paths[cur], pre_region=region)

        res.append(region)

        # report
        logging.getLogger().info(
            '@CHEN-> IOU : -> %.2f <- !!!  score: %.2f for tracking on frame %d, cost %4.2f' \
            % (util.overlap_ratio(gts[cur], region), score, cur, time.time() - T))

        # online update
        scores.append(score)
        if score < 0.125:
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


def track(model, img_path, pre_region):
    feat_bboxes = sample.sample_on_feat()
    pred_data, restore_info = datahelper.get_predict_data(img_path, pre_region, feat_bboxes)
    pred_iter = datahelper.get_predict_iter(pred_data)
    res = model.predict(pred_iter).asnumpy()
    pos_score = res[:, 1]
    topK = 5
    top_idx = pos_score.argsort()[-topK::]

    top_scores = pos_score[top_idx]
    top_feat_bboxes = feat_bboxes[top_idx, 1:]
    top_patch_bboxes = util.feat2img(top_feat_bboxes)

    def restore_img_bbox(opt_patch_bbox, restore_info):
        xo, yo, wo, ho = opt_patch_bbox
        img_W, img_H, X, Y, W, H = restore_info
        x, y = W / 227. * xo + X - img_W, H / 227. * yo + Y - img_H
        w, h = W / 227. * wo, H / 227. * ho

        # CUT in case out of range
        x, y = max(0, x), max(0, y)
        w, h = min(w, img_W - x), min(h, img_H - y)

        return x, y, w, h

    def check_pred_data(i, cur=0):
        img_W, img_H, X, Y, W, H = restore_info
        x, y, w, h = config.gts[cur]
        # x, y = x + img_W - X, y + img_H - Y
        feat_bbox = feat_bboxes[i, 1:].reshape(1, 4)
        patch_bbox = util.feat2img(feat_bbox).reshape(4, )
        img_bbox = restore_img_bbox(patch_bbox, restore_info)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(plt.imread(config.img_paths[cur]))
        ax.add_patch(patches.Rectangle((img_bbox[0], img_bbox[1]), img_bbox[2], img_bbox[3],
                                       linewidth=4, edgecolor='red', facecolor='none'))
        ax.add_patch(patches.Rectangle((x, y), w, h,
                                       linewidth=1, edgecolor='blue', facecolor='none'))
        fig.show()
        return (pos_score[i])

    top_img_bboxes = [restore_img_bbox(x, restore_info) for x in top_patch_bboxes]
    opt_img_bbox = np.mean(top_img_bboxes, 0)
    opt_score = top_scores.mean()

    return opt_img_bbox, opt_score


def debug_track_on_OTB():
    args = parse_args()
    config.ctx = mx.gpu(args.gpu)

    otb = datahelper.OTBHelper(args.OTB_path)
    img_paths, gts = otb.get_seq('Girl')

    # for debug and check
    config.gts = gts
    config.img_paths = img_paths

    model, all_params = extend.init_model(args)

    logging.getLogger().setLevel(logging.DEBUG)
    res, scores = debug_track_seq(args, model, img_paths, gts)


def parse_args():
    parser = argparse.ArgumentParser(description='Train MDNet network')
    parser.add_argument('--gpu', help='GPU device to train with', default=0, type=int)
    parser.add_argument('--num_epoch_for_offline', default=100, type=int)
    parser.add_argument('--num_epoch_for_online', default=0, help='epoch of training for every frame', type=int)
    parser.add_argument('--num_frame_for_offline', default=1, help='epoch of training for every frame', type=int)
    parser.add_argument('--lr_online', help='base learning rate', default=1e-5, type=float)
    parser.add_argument('--wd', help='base learning rate', default=1e-1, type=float)
    parser.add_argument('--OTB_path', help='OTB folder', default='/media/chen/datasets/OTB', type=str)
    parser.add_argument('--VOT_path', help='VOT folder', default='/media/chen/datasets/VOT2015', type=str)
    parser.add_argument('--lr_step', default=121 * 20, help='every 121 num for one epoch', type=int)
    parser.add_argument('--lr_factor', default=0.5, help='20 times will be around 0.1', type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--lr_stop', default=1e-7, type=float)
    parser.add_argument('--lr_offline', default=5e-7, help='base learning rate', type=float)
    parser.add_argument('--fixed_conv', help='the params before(include) which conv are all fixed',
                        default=2, type=int)
    parser.add_argument('--saved_fname', default='conv123', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    debug_track_on_OTB()
