# -*-coding:utf- 8-*-
import Queue
import random

from easydict import EasyDict as edict
import mxnet as mx
import argparse
import numpy as np
import copy
import time

import datahelper
import util
import extend
from nm_suppression import NMSuppression
from setting import const
import matplotlib.pyplot as plt
from matplotlib import patches
import logging
import kit
import os

update_data_queue = Queue.Queue(maxsize=100)
mc_logs = []


def debug_track_seq(args, model, img_paths, gts):
    '''

    :param args:
    :param model:
    :param img_paths: 待跟踪的图片地址list，首个是有标注的，用来首帧训练的。
    :param gts: 用来调试的每帧的gt，本应只传gts[0]
    :return:
    '''

    def check_train(c, pr=None):
        if pr == None:
            pr = gts[c]
        data_batches = datahelper.get_data_batches(datahelper.get_update_data(plt.imread(img_paths[c]), pr, c))
        metric = mx.metric.CompositeEvalMetric()
        metric.add([extend.PR(), extend.RR(), extend.TrackTopKACC(), extend.ACC()])
        metric.reset()
        for data_batch in data_batches:
            model.forward(data_batch, is_train=False)
            model.update_metric(metric, data_batch.label)
        for name, val in metric.get_name_value():
            logging.info('|--| check train %s=%f', name, val)

    train_img_path, train_gt = img_paths[0], gts[0]
    img = plt.imread(train_img_path)
    const.img_H, const.img_W, c = img.shape

    sgd = mx.optimizer.SGD(learning_rate=args.lr_offline, wd=args.wd, momentum=args.momentum)
    sgd.set_lr_mult({'fc4_bias': 2, 'fc5_bias': 2, 'score_bias': 20, 'score_weight': 10})
    model.init_optimizer(kvstore='local', optimizer=sgd, force_init=True)

    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    offline_update(args, model, img, train_gt)

    sgd = mx.optimizer.SGD(learning_rate=args.lr_online, wd=args.wd, momentum=args.momentum)
    sgd.set_lr_mult({'fc4_bias': 2, 'fc5_bias': 2, 'score_bias': 20, 'score_weight': 10})
    model.init_optimizer(kvstore='local', optimizer=sgd, force_init=True)
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

    # res, scores 是保存每一帧的结果位置和给出的是目标的概率的list，包括用来训练的首帧
    res, probs = [gts[0]], [0.8]
    region = gts[0]
    ious = []
    times = []
    last_update = -5
    bh = util.BboxHelper(region)

    def show_res(i):
        kit.show_tracking(plt.imread(img_paths[i]), [res[i]] + [gts[i]])

    # prepare online update data
    add_update_data(img, gts[0], [gts[0], gts[0]])

    length = len(img_paths)
    for cur in range(1, length):
        img = plt.imread(img_paths[cur])
        const.img = img
        img_H, img_W, c = np.shape(img)
        T = time.time()
        # track
        pre_regions = bh.get_base_regions()
        B, P = multi_track(model, img, pre_regions=pre_regions, gt=gts[cur])
        region, prob = util.refine_bbox(B, P, res[-1])

        # logging.info('time for mult-track:%.6f' % (time.time() - T))

        # region = np.mean([region] + res[-2:], 0)

        # logging.info('| cost:%.6f, multi track for %d regions, ' % (time.time() - t, len(pre_regions)))

        def show_tracking():
            gt = gts[cur]
            # estimate_region = pre_regions[1]
            # pre_region = res[cur - 1]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(img)
            ax.add_patch(patches.Rectangle((region[0], region[1]), region[2], region[3],
                                           linewidth=2, edgecolor='red', facecolor='none'))
            ax.add_patch(patches.Rectangle((gt[0], gt[1]), gt[2], gt[3],
                                           linewidth=1, edgecolor='yellow', facecolor='none'))
            # ax.add_patch(patches.Rectangle((pre_region[0], pre_region[1]), pre_region[2], pre_region[3],
            #                                linewidth=1, edgecolor='black', facecolor='none'))
            # ax.add_patch(patches.Rectangle((estimate_region[0], estimate_region[1]),
            #                                estimate_region[2], estimate_region[3],
            #                                linewidth=1, edgecolor='blue', facecolor='none'))
            fig.show()

        def show_BP(th=0.5):
            kit.show_tracking(img, np.array(B)[np.array(P) > th])

        if (prob > 0.6) & (prob > (probs[-1] - 0.1)):
            t = time.time()
            add_update_data(img, region, B)
            # logging.info('time for add data:%.6f' % (time.time() - t))
            if cur - last_update > 10:
                logging.info('| long term update')
                model = online_update(args, model, 15, 5)
                last_update = cur
        else:
            logging.info('| twice tracking %d.jpg for prob: %.6f' % (cur, prob))
            if cur - last_update > 3:
                logging.info('| short term update')
                model = online_update(args, model, 5, 8)
                # last_update = cur

            pre_regions = bh.get_twice_base_regions()
            B, P = multi_track(model, img, pre_regions=pre_regions, gt=gts[cur])
            region, prob = util.refine_bbox(B, P, res[-1])

            if prob < 0.6:
                region = res[-1]
            else:
                add_update_data(img, region, B)
        # if (cur < 10) & (cur % 2 == 0):
        #     model = online_update(args, model, 3, 10)
        #     last_update = cur
        # report
        bh.add_res(region)
        res.append(region)
        probs.append(prob)
        iou = util.overlap_ratio(gts[cur], region)
        cost = time.time() - T
        ious.append(iou)
        times.append(cost)
        logging.info('| IOU : [ %.2f ], prob:%.5f for tracking on frame %d, cost %4.4f' \
                     % (iou, prob, cur, cost))
        # logging.info('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        if iou < 0.5:
            a = 1
        if iou < 0.3:
            a = 1
        if iou == 0:
            a = 1
        if cur % 10 == 0:
            a = 1

    return res, probs, ious


def check_track(model, i, flag=0, pr=None, topK=1):
    if pr == None:
        pr = const.gts[i]

    plotc = 0
    showc = 0
    checkc = 0
    if flag == 1:
        plotc = 1
    if flag == 2:
        showc = 1
    if flag == 3:
        checkc = 1

    bboxes, probs = track(model, plt.imread(const.img_paths[i]), pr, const.gts[i],
                          plotc=plotc, showc=showc, checkc=checkc, topK=topK)
    return probs


def get_update_data(frame_len, batch_num):
    '''
        返回最近frame_len帧 组成的 update_data
    :param frame_len: 长期100，短期20
    :return:
    '''
    # t = time.time()
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
    # logging.info('time for get update data:%.5f' % (time.time() - t))
    return img_patches, feat_bboxes, labels


def add_update_data(img, gt, regions):
    '''
        原版mdnet每一帧采50 pos 200 neg
        返回该帧构造出的 4 个img_patch, each 16 pos 32 neg
    :param img_patch:
    :param gt:
    :return:
    '''
    t = time.time()
    update_data = datahelper.get_update_data(img, gt, regions)
    if update_data_queue.full():
        update_data_queue.get()
    if update_data_queue.empty():
        update_data_queue.put(update_data)
    update_data_queue.put(update_data)
    # logging.info('| add update data, cost:%.6f' % (time.time() - t))


def check_metric(model, data_batches):
    if not const.check_mc:
        return

    metric = mx.metric.CompositeEvalMetric()
    metric.add([extend.PR(), extend.RR(), extend.TrackTopKACC(), mx.metric.CrossEntropy()])
    for data_batch in data_batches:
        model.forward(data_batch, is_train=False)
        model.update_metric(metric, data_batch.label)

    values = metric.get()[1]
    logging.getLogger().info('|----| check metric %.2f,%.2f,%.2f, loss:[%.6f]' %
                             (values[0], values[1], values[2], values[3]))


def check_gt_metric(model, img, gt):
    if not const.check_mc:
        return

    metric = mx.metric.CompositeEvalMetric()
    metric.add([extend.PR(), extend.RR(), extend.TrackTopKACC(), mx.metric.CrossEntropy()])
    data_batches = datahelper.get_data_batches(datahelper.get_train_data(img, gt))
    for data_batch in data_batches:
        model.forward(data_batch, is_train=False)
        model.update_metric(metric, data_batch.label)

    values = metric.get()[1]
    logging.getLogger().info('|----| check gt metric %.2f,%.2f,%.2f, loss:[%.6f]' %
                             (values[0], values[1], values[2], values[3]))


def offline_update(args, model, img, gt):
    logging.info('|------------ offline update -------------')
    data_batches = datahelper.get_data_batches(datahelper.get_train_data(img, gt))
    check_metric(model, data_batches)
    for epoch in range(0, args.num_epoch_for_offline):
        t = time.time()
        model = extend.train_with_hnm(model, data_batches, sel_factor=2)
        check_metric(model, data_batches)
        logging.info('| epoch %d, cost:%.4f, batches: %d ' % (epoch, time.time() - t, len(data_batches)))
        a = 1

    return model


def online_update(args, model, data_len, batch_num):
    logging.info('|-------------- online update -------------')
    data_batches = datahelper.get_data_batches(get_update_data(data_len, batch_num))
    check_metric(model, data_batches)
    for epoch in range(0, args.num_epoch_for_online):
        t = time.time()
        extend.train_with_hnm(model, data_batches, sel_factor=4)
        logging.info('|----| epoch %d, cost:%.4f for %d batches' % (epoch, time.time() - t, len(data_batches)))
        check_metric(model, data_batches)
    return model


def multi_track(model, img, pre_regions, gt, topK=10):
    B, P = [], []

    single_track_topK = 10
    for pr in pre_regions:
        bboxes, probs = track(model, img, pr, gt, topK=single_track_topK)
        B += bboxes
        P += probs

    B = np.array(B)
    P = np.array(P)
    # bbox, idx = NMSuppression(bbs=util.xywh2x1y1x2y2(B), probs=np.array(P),
    #                           overlapThreshold=0.8).fast_suppress()
    #
    # return np.array(B)[idx].tolist(), np.array(P)[idx].tolist()
    top_idx = P.argsort()[-topK::]

    return B[top_idx].tolist(), P[top_idx].tolist()


def track(model, img, pre_region, gt, topK=2, plotc=False, showc=False, checkc=False):
    pred_data, restore_info = datahelper.get_predict_data(img, pre_region)

    pred_iter = datahelper.get_iter(pred_data)
    [img_patch], [feat_bboxes], [l] = pred_data

    # t = time.time()
    res = model.predict(pred_iter).asnumpy()
    # logging.info('| tiem for get predict:%.5f' % (time.time() - t))

    pos_score = res[:, 1]

    patch_bboxes = util.feat2img(feat_bboxes[:, 1:])
    img_bboxes = util.restore_bboxes(patch_bboxes, restore_info)
    labels = util.overlap_ratio(gt, img_bboxes)

    # def nms(th=0.5):
    #     # t = time.time()
    #     bbox, idx = NMSuppression(bbs=util.xywh2x1y1x2y2(img_bboxes), probs=np.array(pos_score),
    #                               overlapThreshold=th).fast_suppress()
    #     # logging.getLogger().info('@CHEN->nms:%.4f' % (time.time() - t))
    #     return idx
    #
    # nms_idx = nms(0.8)
    # top_idx = nms_idx[:topK]

    top_idx = pos_score.argsort()[-topK::]
    top_scores = pos_score[top_idx]
    top_feat_bboxes = feat_bboxes[top_idx, 1:]
    top_patch_bboxes = util.feat2img(top_feat_bboxes)

    top_img_bboxes = util.restore_bboxes(top_patch_bboxes, restore_info)
    opt_img_bbox = np.mean(top_img_bboxes, 0)
    opt_score = top_scores.mean()

    def check_pred_data(i):
        feat_bbox = feat_bboxes[i, 1:].reshape(1, 4)
        patch_bbox = util.feat2img(feat_bbox)
        img_bbox = util.restore_bboxes(patch_bbox, restore_info).reshape(4, )

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img)
        ax.add_patch(patches.Rectangle((img_bbox[0], img_bbox[1]), img_bbox[2], img_bbox[3],
                                       linewidth=4, edgecolor='red', facecolor='none'))
        ax.add_patch(patches.Rectangle((gt[0], gt[1]), gt[2], gt[3],
                                       linewidth=1, edgecolor='blue', facecolor='none'))
        fig.show()
        return (pos_score[i], labels[i])

    def plot():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(pos_score, 'r')
        ax.plot(labels, 'blue')
        fig.show()
        return fig

    def show_tracking():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img)
        ax.add_patch(patches.Rectangle((opt_img_bbox[0], opt_img_bbox[1]), opt_img_bbox[2], opt_img_bbox[3],
                                       linewidth=4, edgecolor='red', facecolor='none'))
        ax.add_patch(patches.Rectangle((gt[0], gt[1]), gt[2], gt[3],
                                       linewidth=1, edgecolor='blue', facecolor='none'))
        ax.add_patch(patches.Rectangle((pre_region[0], pre_region[1]), pre_region[2], pre_region[3],
                                       linewidth=1, edgecolor='yellow', facecolor='none'))
        fig.show()
        return top_img_bboxes, top_scores

    def check_all_metric(th=const.train_pos_th):
        # PR RR
        output_pos_idx = pos_score > th
        hit = np.sum(labels[output_pos_idx] > th)
        PR_len = 1. * np.sum(output_pos_idx)
        RR_len = 1. * np.sum(labels > th)
        # TopK
        topK_idx = pos_score.argsort()[-5::]
        hit2 = np.sum(labels[topK_idx] > th)
        # Loss
        tl = copy.deepcopy(labels)
        tl[tl > const.train_pos_th] = 1
        tl[tl < const.train_neg_th] = 0

        loss = mx.ndarray.softmax_cross_entropy(mx.ndarray.array(res), mx.ndarray.array(tl)).asnumpy()[0]
        logging.getLogger().info('TH_%.1f =>Loss: %6.2f, PR:%.2f, RR:%.2f, TopK:%.2f, IOU:%.2f' % (
            th, loss / labels.shape[0], hit / PR_len, hit / RR_len, hit2 / 5., util.overlap_ratio(gt, opt_img_bbox)))

    def check_train_metric(th=const.train_pos_th):
        # PR RR
        idx = (labels > const.train_pos_th) | (labels < const.train_neg_th)
        import copy
        tl = copy.deepcopy(labels[idx])
        ps = pos_score[idx]
        rs = res[idx, :]

        output_pos_idx = ps > th
        hit = np.sum(tl[output_pos_idx] > th)
        PR_len = 1. * np.sum(output_pos_idx)
        RR_len = 1. * np.sum(tl > th)
        # TopK
        topK_idx = ps.argsort()[-5::]
        hit2 = np.sum(tl[topK_idx] > th)
        # Loss
        tl[tl > const.train_pos_th] = 1
        tl[tl < const.train_neg_th] = 0

        loss = mx.ndarray.softmax_cross_entropy(mx.ndarray.array(rs), mx.ndarray.array(tl)).asnumpy()[0]
        logging.getLogger().info('Check Train =>Loss: %6.2f, PR:%.2f, RR:%.2f, TopK:%.2f, IOU:%.2f' % (
            loss / tl.shape[0], hit / PR_len, hit / RR_len, hit2 / 5., util.overlap_ratio(gt, opt_img_bbox)))

        # show_tracking()

    if plotc:
        plot()
    if showc:
        show_tracking()
    if checkc:
        check_train_metric()
        check_all_metric()
    return top_img_bboxes.tolist(), top_scores.tolist()


def debug_seq():
    args = parse_args()

    vot = datahelper.VOTHelper(args.VOT_path)
    img_paths, gts = vot.get_seq('bolt1')  # gymnastics2

    first_idx = 0
    img_paths, gts = img_paths[first_idx:], gts[first_idx:]
    const.img_H, const.img_W, c = np.shape(plt.imread(img_paths[0]))
    datahelper.get_train_data(plt.imread(img_paths[0]), gts[0])

    # for debug and check
    const.gts = gts
    const.img_paths = img_paths

    model = extend.init_model(args)

    logging.getLogger().setLevel(logging.INFO)
    res, scores, ious = debug_track_seq(args, model, img_paths, gts)
    return


def parse_args():
    parser = argparse.ArgumentParser(description='Train MDNet network')
    parser.add_argument('--gpu', help='GPU device to train with', default=0, type=int)
    parser.add_argument('--num_epoch_for_offline', default=1, type=int)
    parser.add_argument('--num_epoch_for_online', default=1, type=int)

    parser.add_argument('--fixed_conv', default=3, help='these params of [ conv_i <= ? ] will be fixed', type=int)
    parser.add_argument('--saved_fname', default='params/sm_lr_19500/shared', type=str)
    parser.add_argument('--OTB_path', help='OTB folder', default='/media/chen/datasets/OTB', type=str)
    parser.add_argument('--VOT_path', help='VOT folder', default='/home/chen/vot-toolkit/cmdnet-workspace/sequences',
                        type=str)
    parser.add_argument('--ROOT_path', help='cmd folder', default='/home/chen/mx-mdnet', type=str)

    parser.add_argument('--wd', default=1e0, help='weight decay', type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--lr_offline', default=1e-5, help='base learning rate', type=float)
    parser.add_argument('--lr_online', default=3e-5, help='base learning rate', type=float)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    const.check_mc = True
    debug_seq()
