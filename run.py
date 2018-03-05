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
            eva.fit(model, train_img_path=img_paths[i], train_gt=gts[i],
                    val_img_path=img_paths[i], val_pre_region=gts[i], val_gt=gts[i],
                    optimizer='sgd',
                    optimizer_params={'learning_rate': args.lr_offline,
                                      'wd': args.wd,
                                      'momentum': args.momentum,
                                      'clip_gradient': 5,
                                      'lr_scheduler': extend.MDScheduler(
                                          args.lr_step, args.lr_factor, args.lr_stop)},
                    begin_epoch=i * 25, num_epoch=i * 25 + args.num_epoch_for_offline)
    #
    # train_data = datahelper.get_train_data(img_paths[0], gts[0])
    # train_data2 = train_data[0], train_data[1], np.zeros(np.shape(train_data[2]))
    # train_iter = datahelper.get_train_iter(train_data)
    # train_iter2 = datahelper.get_train_iter(train_data2)
    # label = np.reshape(train_data[2], -1)

    # import eva
    # res = eva.predict(model.symbol, model.get_params()[0], train_data)
    # res1 = model.predict(train_iter).asnumpy()
    # res2 = model.predict(train_iter2).asnumpy()
    # r2 = (res2 * 2) ** 0.5

    #
    # a, b = model.get_params()
    # mx.ndarray.save('params/weighted_by_' + str(args.weight_factor) + '_fixed' + str(args.fixed_conv), a)
    # exit()

    res = []
    scores = []
    length = len(img_paths)
    region = gts[0]

    for cur in range(0, length):
        T = time.time()
        # track
        region, score = track(copy.deepcopy(model.get_params()[0]), img_paths[cur], pre_region=region)

        res.append(region)

        # report
        logging.getLogger().info(
            '@CHEN-> IOU : %.2f !!!  s: %.2f for tracking on frame %d, cost %6.2f' \
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


def train_on_first(args, model, first_path, gt, begin_epoch=0, num_epoch=100):
    return model


def track(arg_params, img_path, pre_region):
    # only for iou loss
    feat_bboxes = sample.sample_on_feat()
    pred_data, restore_info = datahelper.get_predict_data(img_path, pre_region, feat_bboxes)

    import eva
    res = eva.predict(arg_params, pred_data[0], pred_data[1])

    def restore_img_bbox(opt_patch_bbox, restore_info):
        xo, yo, wo, ho = opt_patch_bbox
        img_W, img_H, X, Y, W, H = restore_info
        x, y = W / 227. * xo + X - img_W, H / 227. * yo + Y - img_H
        w, h = W / 227. * wo, H / 227. * ho

        # CUT in case out of range
        x, y = max(0, x), max(0, y)
        w, h = min(w, img_W - x), min(h, img_H - y)

        return x, y, w, h

    opt_idx = mx.ndarray.topk(res, k=10).asnumpy().astype('int32')
    res = res.asnumpy()
    opt_scores = res[opt_idx]
    opt_score = opt_scores.mean()
    opt_feat_bboxes = feat_bboxes[opt_idx, 1:]
    opt_patch_bboxes = util.feat2img(opt_feat_bboxes)
    opt_patch_bbox = opt_patch_bboxes.mean(0)
    opt_img_bbox = restore_img_bbox(opt_patch_bbox, restore_info)

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
                                       linewidth=2, edgecolor='red', facecolor='none'))
        ax.add_patch(patches.Rectangle((x, y), w, h,
                                       linewidth=2, edgecolor='blue', facecolor='none'))
        fig.show()
        return (res[i])

    return opt_img_bbox, opt_score


def debug_track_on_OTB():
    args = parse_args()
    config.ctx = mx.gpu(args.gpu)

    otb = datahelper.OTBHelper(args.OTB_path)
    img_paths, gts = otb.get_seq('Liquor')

    # for debug and check
    config.gts = gts
    config.img_paths = img_paths

    model, all_params = extend.init_model(args)

    logging.getLogger().setLevel(logging.DEBUG)
    res, scores = debug_track_seq(args, model, img_paths, gts)


def parse_args():
    parser = argparse.ArgumentParser(description='Train MDNet network')
    parser.add_argument('--loss_type', type=int, default=3,
                        help='0 for {0,1} corss-entropy, 1 for smooth_l1, 2 for {pos_pred} corss-entropy,'
                             '3 for CE loss and weighted for high iou label')
    parser.add_argument('--gpu', help='GPU device to train with', default=0, type=int)
    parser.add_argument('--num_epoch_for_offline', default=200, type=int)
    parser.add_argument('--num_epoch_for_online', default=0, help='epoch of training for every frame', type=int)
    parser.add_argument('--num_frame_for_offline', default=1, help='epoch of training for every frame', type=int)
    parser.add_argument('--batch_callback_freq', default=50, type=int)
    parser.add_argument('--lr_online', help='base learning rate', default=1e-5, type=float)
    parser.add_argument('--wd', help='base learning rate', default=1e-1, type=float)
    parser.add_argument('--OTB_path', help='OTB folder', default='/media/chen/datasets/OTB', type=str)
    parser.add_argument('--VOT_path', help='VOT folder', default='/media/chen/datasets/VOT2015', type=str)
    parser.add_argument('--p_level', help='print level, default is 0 for debug mode', default=0, type=int)
    parser.add_argument('--lr_step', default=36 * 50, help='every 36 num for one epoch', type=int)
    parser.add_argument('--lr_factor', default=0.5, help='20 times will be around 0.1', type=float)
    parser.add_argument('--lr_stop', default=5e-8, type=float)
    parser.add_argument('--iou_acc_th', default=0.1, type=float)
    parser.add_argument('--momentum', default=0, type=float)
    parser.add_argument('--log', default=1, type=int)

    parser.add_argument('--lr_offline', default=5e-8, help='base learning rate', type=float)
    parser.add_argument('--weight_factor', default=10, type=float)
    parser.add_argument('--fixed_conv', help='the params before(include) which conv are all fixed',
                        default=1, type=int)
    parser.add_argument('--saved_fname', default='conv123', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    debug_track_on_OTB()
