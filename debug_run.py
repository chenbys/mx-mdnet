# -*-coding:utf- 8-*-
import Queue

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

update_data_queue = Queue.Queue(maxsize=100)


def debug_track_seq(args, model, img_paths, gts):
    '''

    :param args:
    :param model:
    :param img_paths: 待跟踪的图片地址list，首个是有标注的，用来首帧训练的。
    :param gts: 用来调试的每帧的gt，本应只传gts[0]
    :return:
    '''
    train_img_path, train_gt = img_paths[0], gts[0]
    img = plt.imread(train_img_path)
    train_iter = datahelper.get_iter(datahelper.get_train_data(img, train_gt))
    eval_iter = datahelper.get_iter(datahelper.get_train_data(plt.imread(img_paths[5]), gts[5]))

    model.fit(train_data=train_iter, eval_data=eval_iter, optimizer='sgd',
              eval_metric=mx.metric.CompositeEvalMetric(
                  [extend.SMLoss(), extend.PR(0.5), extend.RR(0.5), extend.TrackTopKACC(10, 0.6)]),
              optimizer_params={'learning_rate': args.lr_offline, 'wd': args.wd, 'momentum': args.momentum,
                                'lr_scheduler': mx.lr_scheduler.FactorScheduler(step=args.lr_step,
                                                                                factor=args.lr_factor,
                                                                                stop_factor_lr=args.lr_stop),
                                # 'clip_gradient': 5,
                                },
              begin_epoch=0, num_epoch=args.num_epoch_for_offline)

    # res, scores 是保存每一帧的结果位置和给出的是目标的概率的list，包括用来训练的首帧
    res, probs = [gts[0]], [1]
    region = gts[0]
    ious = []

    # prepare online update data
    add_update_data(img, gts[0])

    length = len(img_paths)
    for cur in range(1, length):
        img = plt.imread(img_paths[cur])
        img_H, img_W, c = np.shape(img)
        T = time.time()
        # track
        pre_region = region
        pre_regions = []
        for dx, dy, ws, hs in [[0, 0, 1, 1],
                               [0, 0, 2, 2],
                               [0, 0, 0.5, 0.5]]:
            pre_regions.append(util.central_bbox(pre_region, dx, dy, ws, hs, img_W, img_H))

        region, prob = multi_track(model, img, pre_regions=pre_regions, gt=gts[cur])

        def show_tracking():
            gt = gts[cur]
            pre_region = res[cur - 1]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(img)
            ax.add_patch(patches.Rectangle((region[0], region[1]), region[2], region[3],
                                           linewidth=3, edgecolor='red', facecolor='none'))
            ax.add_patch(patches.Rectangle((gt[0], gt[1]), gt[2], gt[3],
                                           linewidth=1, edgecolor='blue', facecolor='none'))
            ax.add_patch(patches.Rectangle((pre_region[0], pre_region[1]), pre_region[2], pre_region[3],
                                           linewidth=1, edgecolor='yellow', facecolor='none'))
            fig.show()

        # twice tracking
        if prob > 0.5:
            add_update_data(img, region)

            if cur % 10 == 0:
                logging.getLogger().info('@CHEN->long term update')
                model = online_update(args, model, 100)
        else:
            logging.getLogger().info('@CHEN->Short term update and Twice tracking')
            model = online_update(args, model, 30)
            pre_region = res[cur - 1]
            # 二次检测时，检查上上次的pre_region，并搜索更大的区域
            pre_regions = res[-7:]
            for dx in [0]:
                for dy in [0]:
                    for ws in [0.5, 0.7, 1, 1.5, 2]:
                        for hs in [0.5, 0.7, 1, 1.5, 2]:
                            pre_regions.append(util.central_bbox(pre_region, dx, dy, ws, hs, img_W, img_H))

            region, prob = multi_track(model, img, pre_regions=pre_regions, gt=gts[cur])

            if prob > 0.5:
                add_update_data(img, region)

        # report
        res.append(region)
        probs.append(prob)
        iou = util.overlap_ratio(gts[cur], region)
        ious.append(iou)
        logging.getLogger().info(
            '@CHEN-> IOU : [ %.2f ] !!!  prob: %.2f for tracking on frame %d, cost %4.4f' \
            % (iou, prob, cur, time.time() - T))

        next_frame = 1

    return res, probs, ious


def check_track(model, i, plotc=False, showc=False, checkc=False):
    bboxes, probs = track(model, plt.imread(const.img_paths[i]), const.gts[i], const.gts[i],
                          plotc=plotc, showc=showc, checkc=checkc)
    return probs


def get_update_data(frame_len=20):
    '''
        返回最近frame_len帧 组成的 update_data
    :param frame_len: 长期100，短期20
    :return:
    '''
    frame_len = min(frame_len, update_data_queue.qsize())
    img_patches, feat_bboxes, labels = [], [], []

    for i in range(1, frame_len):
        a, b, c = update_data_queue.queue[-(i % frame_len)]
        img_patches += a
        feat_bboxes += b
        labels += c

    return img_patches, feat_bboxes, labels


def add_update_data(img, gt):
    '''
        原版mdnet每一帧采50 pos 200 neg
        返回该帧构造出的 4 个img_patch, each 16 pos 32 neg
    :param img_patch:
    :param gt:
    :return:
    '''
    update_data = datahelper.get_update_data(img, gt)
    if update_data_queue.full():
        update_data_queue.get()
    if update_data_queue.empty():
        update_data_queue.put(update_data)
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
    update_iter = datahelper.get_iter(get_update_data(data_len))

    model.fit(train_data=update_iter, optimizer='sgd',
              eval_metric=mx.metric.CompositeEvalMetric(
                  [extend.SMLoss(), extend.PR(0.5), extend.RR(0.5), extend.TrackTopKACC(10, 0.6)]),
              optimizer_params={'learning_rate': args.lr_offline,
                                'wd': args.wd,
                                'momentum': args.momentum,
                                # 'clip_gradient': 5,
                                },
              begin_epoch=0, num_epoch=args.num_epoch_for_online)
    return model


def multi_track(model, img, pre_regions, gt, topK=5):
    A, B = [], []
    for pr in pre_regions:
        t = time.time()
        bboxes, probs = track(model, img, pr, gt, topK=topK)
        A.append(bboxes)
        B.append(probs)
        # print 'time for track:%.5f' % (time.time() - t)
    idx = np.array(B).reshape(-1, ).argsort()[-16::]
    x_y_idx = [divmod(i, topK) for i in idx]
    top_bboxes = []
    top_probs = []
    for x, y in x_y_idx:
        top_bboxes.append(A[x][y, :])
        top_probs.append(B[x][y])
    opt_img_bbox = np.mean(top_bboxes, 0)
    opt_score = np.mean(top_probs)

    def show_tracking():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img)
        ax.add_patch(patches.Rectangle((opt_img_bbox[0], opt_img_bbox[1]), opt_img_bbox[2], opt_img_bbox[3],
                                       linewidth=4, edgecolor='red', facecolor='none'))
        ax.add_patch(patches.Rectangle((gt[0], gt[1]), gt[2], gt[3],
                                       linewidth=1, edgecolor='blue', facecolor='none'))
        fig.show()

    return opt_img_bbox, opt_score


def track(model, img, pre_region, gt, topK=5, plotc=False, showc=False, checkc=False):
    pred_data, restore_info = datahelper.get_predict_data(img, pre_region)
    pred_iter = datahelper.get_iter(pred_data)
    [img_patch], [feat_bboxes], [l] = pred_data
    res = model.predict(pred_iter).asnumpy()
    pos_score = res[:, 1]

    patch_bboxes = util.feat2img(feat_bboxes[:, 1:])
    img_bboxes = util.restore_bboxes(patch_bboxes, restore_info)
    labels = util.overlap_ratio(gt, img_bboxes)
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
        plt.plot(pos_score, 'r')
        plt.plot(labels, 'blue')

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

    def nms():
        t = time.time()
        bbox, idx = NMSuppression(bbs=top_img_bboxes, probs=np.array(top_scores), overlapThreshold=0.5).fast_suppress()
        logging.getLogger().info('@CHEN->get:%.4f' % (time.time() - t))
        return idx

    # check_all_metric(0.5)
    # check_all_metric(0.7)
    # check_train_metric()
    if plotc:
        plot()
    if showc:
        show_tracking()
    if checkc:
        check_train_metric()
        check_all_metric()

    return top_img_bboxes, top_scores


# track(model,plt.imread(const.img_paths[0]),const.gts[0],const.gts[0])


def debug_seq():
    args = parse_args()

    vot = datahelper.VOTHelper(args.VOT_path)
    img_paths, gts = vot.get_seq('bag')

    first_idx = 0
    img_paths, gts = img_paths[first_idx:], gts[first_idx:]

    # for debug and check
    const.gts = gts
    const.img_paths = img_paths

    model, all_params = extend.init_model(args)

    logging.getLogger().setLevel(logging.INFO)
    res, scores, ious = debug_track_seq(args, model, img_paths, gts)
    return


def parse_args():
    parser = argparse.ArgumentParser(description='Train MDNet network')
    parser.add_argument('--gpu', help='GPU device to train with', default=0, type=int)
    parser.add_argument('--num_epoch_for_offline', default=5, type=int)
    parser.add_argument('--num_epoch_for_online', default=1, type=int)

    parser.add_argument('--fixed_conv', help='these params of [ conv_i <= ? ] will be fixed', default=3, type=int)
    parser.add_argument('--saved_fname', default='conv123fc4fc5', type=str)
    parser.add_argument('--OTB_path', help='OTB folder', default='/media/chen/datasets/OTB', type=str)
    parser.add_argument('--VOT_path', help='VOT folder', default='/media/chen/datasets/VOT2015', type=str)
    parser.add_argument('--ROOT_path', help='cmd folder', default='/home/chen/mx-mdnet', type=str)

    parser.add_argument('--lr_step', default=307 * 2, help='every x num for y epoch', type=int)
    parser.add_argument('--lr_factor', default=0.8, help='20 times will be around 0.1', type=float)
    parser.add_argument('--lr_stop', default=1e-5, type=float)

    parser.add_argument('--wd', default=5e0, help='weight decay', type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--lr_offline', default=2e-5, help='base learning rate', type=float)
    parser.add_argument('--lr_online', default=1e-5, help='base learning rate', type=float)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    debug_seq()
