# -*-coding:utf- 8-*-

import Queue
import mxnet as mx
import argparse
import numpy as np
import datahelper
import util
import extend

update_data_queue = Queue.Queue(maxsize=100)


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
    for i in range(1, frame_len + 1, step):
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
    model.fit(train_data=update_iter, optimizer='sgd', begin_epoch=0, num_epoch=args.num_epoch_for_online,
              eval_metric=mx.metric.CompositeEvalMetric(
                  [extend.SMLoss(), extend.PR(0.7), extend.RR(0.7), extend.TrackTopKACC(10, 0.7)]),
              optimizer_params={'learning_rate': args.lr_offline,
                                'wd': args.wd, 'momentum': args.momentum})
    return model


def multi_track(model, img, pre_regions, topK=5):
    A, B = [], []
    for pr in pre_regions:
        bboxes, probs = track(model, img, pr, topK=topK)
        A.append(bboxes)
        B.append(probs)
    idx = np.array(B).reshape(-1, ).argsort()[-10::]
    x_y_idx = [divmod(i, topK) for i in idx]
    top_bboxes = []
    top_probs = []
    for x, y in x_y_idx:
        top_bboxes.append(A[x][y, :])
        top_probs.append(B[x][y])
    opt_img_bbox = np.mean(top_bboxes, 0)
    opt_score = np.mean(top_probs)

    opt_img_bbox[2] = max(opt_img_bbox[2], 10)
    opt_img_bbox[3] = max(opt_img_bbox[3], 10)

    return opt_img_bbox, opt_score


def track(model, img, pre_region, topK=5):
    pred_data, restore_info = datahelper.get_predict_data(img, pre_region)

    pred_iter = datahelper.get_iter(pred_data)
    [img_patch], [feat_bboxes], [labels] = pred_data

    res = model.predict(pred_iter).asnumpy()
    pos_score = res[:, 1]

    top_idx = pos_score.argsort()[-topK::]
    top_scores = pos_score[top_idx]
    top_feat_bboxes = feat_bboxes[top_idx, 1:]
    top_patch_bboxes = util.feat2img(top_feat_bboxes)
    top_img_bboxes = util.restore_img_bbox(top_patch_bboxes, restore_info)

    opt_img_bbox = np.mean(top_img_bboxes, 0)
    opt_score = top_scores.mean()

    return top_img_bboxes, top_scores


def parse_args():
    parser = argparse.ArgumentParser(description='Train MDNet network')
    parser.add_argument('--gpu', help='GPU device to train with', default=0, type=int)
    parser.add_argument('--num_epoch_for_offline', default=5, type=int)
    parser.add_argument('--num_epoch_for_online', default=1, type=int)
    parser.add_argument('--fixed_conv', help='these params of [ conv_i <= ? ] will be fixed', default=3, type=int)
    parser.add_argument('--saved_fname', default='conv123fc4fc5', type=str)

    parser.add_argument('--ROOT_path', help='cmd folder', default='/home/chen/mx-mdnet', type=str)
    parser.add_argument('--lr_step', default=307 * 2, help='every 121 num for one epoch', type=int)
    parser.add_argument('--lr_factor', default=0.8, help='20 times will be around 0.1', type=float)
    parser.add_argument('--lr_stop', default=1e-5, type=float)

    parser.add_argument('--wd', default=5e0, help='weight decay', type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--lr_offline', default=2e-5, help='base learning rate', type=float)
    parser.add_argument('--lr_online', default=1e-5, help='base learning rate', type=float)

    args = parser.parse_args()
    return args
