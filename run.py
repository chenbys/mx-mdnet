# -*-coding:utf- 8-*-

import Queue
import mxnet as mx
import argparse
import numpy as np
import datahelper
import util
import extend
from nm_suppression import NMSuppression

update_data_queue = Queue.Queue(maxsize=100)


def get_update_data(frame_len, step):
    '''
        返回最近frame_len帧 组成的 update_data
    :param frame_len: 长期100，短期20
    :return:
    '''
    frame_len = min(frame_len, update_data_queue.qsize())
    img_patches, feat_bboxes, labels = [], [], []

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
    update_data = datahelper.get_update_data(img, gt, cur)
    if update_data_queue.full():
        update_data_queue.get()
    if update_data_queue.empty():
        update_data_queue.put(update_data)
    update_data_queue.put(update_data)


def online_update(args, model, data_len, step):
    '''

    :param args:
    :param model:
    :param img_paths:
    :param res:
    :param cur:
    :param history_len:
    :param num_epoch:
    :return:
    '''
    data_batches = datahelper.get_data_batches(get_update_data(data_len, step))
    batch_mc = extend.SMLoss()
    for epoch in range(0, args.num_epoch_for_online):
        batch_idx = np.arange(0, len(data_batches))
        while True:
            batch_mc_log = []

            for idx in batch_idx:
                batch_mc.reset()
                data_batch = data_batches[idx]
                model.forward_backward(data_batch)
                model.update()
                model.update_metric(batch_mc, data_batch.label)

                batch_mc_log.append(batch_mc.get_name_value()[0][1])

            # 找出最大loss的多更新
            sorted_idx = np.argsort(batch_mc_log)
            sel = len(batch_idx) * 1 / 4
            if sel < len(data_batches) / 20:
                break
            batch_idx = batch_idx[sorted_idx[-sel:]]

    return model


def multi_track(model, img, pre_regions, topK=3):
    B, P = [], []

    single_track_topK = 2
    for pr in pre_regions:
        bboxes, probs = track(model, img, pr, topK=single_track_topK)
        B += bboxes
        P += probs

    return B, P


def track(model, img, pre_region, topK=2):
    pred_data, restore_info = datahelper.get_predict_data(img, pre_region)

    pred_iter = datahelper.get_iter(pred_data)
    [img_patch], [feat_bboxes], [labels] = pred_data

    res = model.predict(pred_iter).asnumpy()
    pos_score = res[:, 1]
    patch_bboxes = util.feat2img(feat_bboxes[:, 1:])
    img_bboxes = util.restore_bboxes(patch_bboxes, restore_info)

    def nms(th=0.85):
        # t = time.time()
        bbox, idx = NMSuppression(bbs=util.xywh2x1y1x2y2(img_bboxes), probs=np.array(pos_score),
                                  overlapThreshold=th).fast_suppress()
        # logging.getLogger().info('@CHEN->nms:%.4f' % (time.time() - t))
        return idx

    nms_idx = nms(0.7)
    top_idx = nms_idx[:topK]
    return img_bboxes[top_idx, :].tolist(), pos_score[top_idx].tolist()


def parse_args():
    parser = argparse.ArgumentParser(description='Train MDNet network')
    parser.add_argument('--gpu', help='GPU device to train with', default=0, type=int)
    parser.add_argument('--num_epoch_for_offline', default=5, type=int)
    parser.add_argument('--num_epoch_for_online', default=1, type=int)

    parser.add_argument('--fixed_conv', default=3, help='these params of [ conv_i <= ? ] will be fixed', type=int)
    parser.add_argument('--saved_fname', default='conv123fc4fc5', type=str)
    parser.add_argument('--OTB_path', help='OTB folder', default='/media/chen/datasets/OTB', type=str)
    parser.add_argument('--VOT_path', help='VOT folder', default='/home/chen/vot-toolkit/cmdnet-workspace/sequences',
                        type=str)
    parser.add_argument('--ROOT_path', help='cmd folder', default='/home/chen/mx-mdnet', type=str)

    parser.add_argument('--wd', default=2e0, help='weight decay', type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--lr_offline', default=1e-5, help='base learning rate', type=float)
    parser.add_argument('--lr_online', default=4e-5, help='base learning rate', type=float)

    args = parser.parse_args()
    return args
