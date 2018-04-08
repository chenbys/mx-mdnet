# -*-coding:utf- 8-*-
import Queue
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
        metric.add([extend.PR(), extend.RR(), extend.TrackTopKACC(), extend.SMLoss()])
        metric.reset()
        for data_batch in data_batches:
            model.forward(data_batch, is_train=False)
            model.update_metric(metric, data_batch.label)
        for name, val in metric.get_name_value():
            logging.info('|--| check train %s=%f', name, val)

    train_img_path, train_gt = img_paths[0], gts[0]
    img = plt.imread(train_img_path)

    sgd = mx.optimizer.SGD(learning_rate=args.lr_offline, wd=args.wd, momentum=args.momentum)
    sgd.set_lr_mult({'score_bias': 10, 'score_weight': 10})
    model.init_optimizer(kvstore='local', optimizer=sgd, force_init=True)

    data_batches = datahelper.get_data_batches(datahelper.get_train_data(img, train_gt))
    check_metric(model, data_batches)
    logging.info('@CHEN->update %3d.' % len(data_batches))
    epoch_mc = mx.metric.CompositeEvalMetric()
    epoch_mc.add([extend.PR(), extend.RR(), extend.TrackTopKACC(), extend.SMLoss()])
    batch_mc = extend.SMLoss()
    mc_log = []
    for epoch in range(0, args.num_epoch_for_offline):
        t = time.time()
        batch_idx = np.arange(0, len(data_batches))
        epoch_mc.reset()
        while True:
            batch_mc_log = []

            for idx in batch_idx:
                batch_mc.reset()
                data_batch = data_batches[idx]
                model.forward_backward(data_batch)
                model.update()
                model.update_metric(batch_mc, data_batch.label)
                model.update_metric(epoch_mc, data_batch.label)

                batch_mc_log.append(batch_mc.get_name_value()[0][1])

            # 找出最大loss的多更新
            sorted_idx = np.argsort(batch_mc_log)
            sel = len(batch_idx) - len(data_batches) / 5
            if sel < len(data_batches) / 10:
                break
            batch_idx = batch_idx[sorted_idx[-sel:]]

        for name, val in epoch_mc.get_name_value():
            logging.info('|-- %s=%f', name, val)

        logging.info('| epoch %d, cost:%.4f' % (epoch, time.time() - t))
    mc_logs.append(mc_log)

    sgd = mx.optimizer.SGD(learning_rate=args.lr_online, wd=args.wd, momentum=args.momentum)
    sgd.set_lr_mult({'score_bias': 10, 'score_weight': 10})
    model.init_optimizer(kvstore='local', optimizer=sgd, force_init=True)

    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    # res, scores 是保存每一帧的结果位置和给出的是目标的概率的list，包括用来训练的首帧
    res, probs = [gts[0]], [1]
    region = gts[0]
    ious = []
    times = []

    def show_res(i):
        kit.show_tracking(plt.imread(img_paths[i]), res[i])

    # prepare online update data
    add_update_data(img, gts[0], 0)

    length = len(img_paths)
    for cur in range(1, length):
        img = plt.imread(img_paths[cur])
        img_H, img_W, c = np.shape(img)
        T = time.time()
        # track
        pre_region = region
        pre_regions = []
        for dx, dy, ws, hs in [
            [0, 0, 1, 1],
        ]:
            pre_regions.append(util.central_bbox(pre_region, dx, dy, ws, hs, img_W, img_H))
        pre_regions += util.replace_wh(region, res[-5:-6] + res[-2:-1])
        t = time.time()
        region, prob = multi_track(model, img, pre_regions=pre_regions, gt=gts[cur])

        # region = np.mean([region] + res[-2:], 0)

        # logging.info('| cost:%.6f, multi track for %d regions, ' % (time.time() - t, len(pre_regions)))

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

        # print util.overlap_ratio(region, gts[cur]), prob
        # twice tracking
        if cur == 57:
            a = 1
        if (prob > 0.6) & (prob > (probs[-1] - 0.2)):

            add_update_data(img, region, cur)

            if cur % 10 == 0:
                if cur < 50:
                    on_len = 50
                else:
                    on_len = 20
                logging.info('| long term update')
                model = online_update(args, model, on_len)
        else:
            logging.info('| short term update for porb: %.2f' % prob)
            model = online_update(args, model, 10)
            logging.info('| twice tracking %d.jpg' % cur)

            pre_region = res[cur - 1]
            # 二次检测时，检查上上次的pre_region，并搜索更大的区域
            pre_regions = util.replace_wh(region, res[-15:-1:3])

            for ws, hs in zip([0.8, 1.5, 1],
                              [0.8, 1.5, 1]):
                pre_regions.append(util.central_bbox(pre_region, 0, 0, ws, hs, img_W, img_H))

            region, prob = multi_track(model, img, pre_regions=pre_regions, gt=gts[cur])

            if prob > 0.6:
                add_update_data(img, region, cur)

        # report
        res.append(region)
        probs.append(prob)
        iou = util.overlap_ratio(gts[cur], region)
        cost = time.time() - T
        ious.append(iou)
        times.append(cost)
        logging.info('| IOU : [ %.2f ], prob:%.5f for tracking on frame %d, cost %4.4f' \
                     % (iou, prob, cur, cost))
        logging.info('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        if iou < 0.4:
            a = 1
        if iou == 0:
            a = 1

    return res, probs, ious


def check_track(model, i, flag=0, pr=None, topK=3):
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


def get_update_data(frame_len=20):
    '''
        返回最近frame_len帧 组成的 update_data
    :param frame_len: 长期100，短期20
    :return:
    '''
    frame_len = min(frame_len, update_data_queue.qsize())
    img_patches, feat_bboxes, labels = [], [], []

    if update_data_queue.qsize() > 20:
        step = 2
    else:
        step = 1
    for i in range(1, frame_len + 2, step):
        a, b, c = update_data_queue.queue[-(i % frame_len)]
        img_patches += a
        feat_bboxes += b
        labels += c
    return img_patches, feat_bboxes, labels


def add_update_data(img, gt, cur):
    '''
        原版mdnet每一帧采50 pos 200 neg
        返回该帧构造出的 4 个img_patch, each 16 pos 32 neg
    :param img_patch:
    :param gt:
    :return:
    '''
    # t = time.time()
    update_data = datahelper.get_update_data(img, gt, cur)
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
    metric.add([extend.PR(), extend.RR(), extend.TrackTopKACC(), extend.SMLoss()])
    metric.reset()
    for data_batch in data_batches:
        model.forward(data_batch, is_train=False)
        model.update_metric(metric, data_batch.label)
    for name, val in metric.get_name_value():
        logging.info('|--| check metric %s=%f', name, val)


def online_update(args, model, data_len):
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
    t = time.time()
    data_batches = datahelper.get_data_batches(get_update_data(data_len))
    check_metric(model, data_batches)
    epoch_mc = mx.metric.CompositeEvalMetric()
    epoch_mc.add([extend.PR(), extend.RR(), extend.TrackTopKACC(), extend.SMLoss()])
    batch_mc = extend.SMLoss()
    for epoch in range(0, args.num_epoch_for_online):
        t = time.time()
        batch_idx = np.arange(0, len(data_batches))
        epoch_mc.reset()
        while True:
            batch_mc_log = []

            for idx in batch_idx:
                batch_mc.reset()
                data_batch = data_batches[idx]
                model.forward_backward(data_batch)
                model.update()
                model.update_metric(batch_mc, data_batch.label)
                model.update_metric(epoch_mc, data_batch.label)

                batch_mc_log.append(batch_mc.get_name_value()[0][1])

            # 找出最大loss的多更新
            sorted_idx = np.argsort(batch_mc_log)
            sel = len(batch_idx) - len(data_batches) / 5
            if sel < len(data_batches) / 10:
                break
            batch_idx = batch_idx[sorted_idx[-sel:]]

        for name, val in epoch_mc.get_name_value():
            logging.info('|-- %s=%f', name, val)

        logging.info('| epoch %d, cost:%.4f' % (epoch, time.time() - t))

    check_metric(model, data_batches)
    return model


def multi_track(model, img, pre_regions, gt, topK=3):
    def show_tracking():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img)
        ax.add_patch(patches.Rectangle((opt_img_bbox[0], opt_img_bbox[1]), opt_img_bbox[2], opt_img_bbox[3],
                                       linewidth=4, edgecolor='red', facecolor='none'))
        ax.add_patch(patches.Rectangle((gt[0], gt[1]), gt[2], gt[3],
                                       linewidth=1, edgecolor='blue', facecolor='none'))
        fig.show()

    A, B = [], []

    # bboxes, probs = track(model, img, pre_regions[0], gt, topK=3)
    # A.append(bboxes)
    # B.append(probs)
    # if np.mean(probs) > 0.7:
    #     opt_img_bbox = np.mean(bboxes, 0)
    #     opt_score = np.mean(probs)
    #     logging.getLogger().info('| once hit')
    #     return opt_img_bbox, opt_score
    single_track_topK = topK * 2
    for pr in pre_regions:
        # t = time.time()
        bboxes, probs = track(model, img, pr, gt, topK=single_track_topK)
        A.append(bboxes)
        B.append(probs)
        # logging.info('| %.6f， time for track' % (time.time() - t))
    idx = np.array(B).reshape(-1, ).argsort()[-topK * len(pre_regions)::]
    x_y_idx = [divmod(i, single_track_topK) for i in idx]
    top_bboxes = []
    top_probs = []
    for x, y in x_y_idx:
        top_bboxes.append(A[x][y, :])
        top_probs.append(B[x][y])

    util.refine_bbox(top_bboxes, top_probs, pre_regions[0])
    opt_img_bbox = np.mean(top_bboxes, 0)
    opt_score = np.mean(top_probs)

    return opt_img_bbox, opt_score


def track(model, img, pre_region, gt, topK=3, plotc=False, showc=False, checkc=False):
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


def debug_seq():
    args = parse_args()

    vot = datahelper.VOTHelper(args.VOT_path)
    img_paths, gts = vot.get_seq('bolt2')

    first_idx = 0
    img_paths, gts = img_paths[first_idx:], gts[first_idx:]

    # for debug and check
    const.gts = gts
    const.img_paths = img_paths

    # datahelper.get_predict_data(plt.imread(img_paths[12]), gts[11])
    model, all_params = extend.init_model(args)

    logging.getLogger().setLevel(logging.INFO)
    res, scores, ious = debug_track_seq(args, model, img_paths, gts)
    return


def parse_args():
    parser = argparse.ArgumentParser(description='Train MDNet network')
    parser.add_argument('--gpu', help='GPU device to train with', default=0, type=int)
    parser.add_argument('--num_epoch_for_offline', default=10, type=int)
    parser.add_argument('--num_epoch_for_online', default=1, type=int)

    parser.add_argument('--fixed_conv', help='these params of [ conv_i <= ? ] will be fixed', default=3, type=int)
    parser.add_argument('--saved_fname', default='conv123fc4fc5', type=str)
    parser.add_argument('--OTB_path', help='OTB folder', default='/media/chen/datasets/OTB', type=str)
    parser.add_argument('--VOT_path', help='VOT folder', default='/media/chen/datasets/VOT2015', type=str)
    parser.add_argument('--ROOT_path', help='cmd folder', default='/home/chen/mx-mdnet', type=str)

    parser.add_argument('--wd', default=1e0, help='weight decay', type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--lr_offline', default=6e-6, help='base learning rate', type=float)
    parser.add_argument('--lr_online', default=2e-5, help='base learning rate', type=float)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    t = time.time()
    # a = datahelper.get_update_data(plt.imread('/media/chen/datasets/OTB/Liquor/img/0001.jpg'), [256, 152, 73, 210])
    print time.time() - t
    t = time.time()
    a = datahelper.get_update_data(plt.imread('/media/chen/datasets/OTB/Liquor/img/0001.jpg'), [256, 152, 73, 210], 30)
    print time.time() - t
    const.check_mc = True
    debug_seq()
