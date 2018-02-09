import mxnet as mx
import argparse
import numpy as np
import sample
import datahelper
import util
import extend
from setting import config
import matplotlib.pyplot as plt
from matplotlib import patches
import logging


def track_seq(model, img_paths, first_gt):
    # train on first frame
    logging.getLogger().setLevel(logging.DEBUG)

    model = train_on_first(model, img_paths[0], first_gt)

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
        print score
        scores.append(score)
        if score < 0:
            # short term update
            model = online_update(model, img_paths, res, cur)
        elif cur % 10 == 0:
            # long term update
            model = online_update(model, img_paths, res, cur)

    return res, scores


def online_update(args, model, img_paths, res, cur, history_len=10, num_epoch=10):
    for i in range(max(0, cur - history_len), cur + 1):
        metric = mx.metric.CompositeEvalMetric()
        metric.add(extend.MDNetIOUACC())
        metric.add(extend.MDNetIOULoss())
        train_iter = datahelper.get_train_iter(datahelper.get_train_data(img_paths[i], res[i]))
        model.fit(train_data=train_iter, optimizer='sgd',
                  optimizer_params={'learning_rate': args.lr_online,
                                    'wd'           : args.wd,
                                    'momentum'     : args.momentum,
                                    'clip_gradient': 5,
                                    'lr_scheduler' : mx.lr_scheduler.FactorScheduler(
                                        args.lr_step, args.lr_factor, args.lr_stop)},
                  eval_metric=metric, begin_epoch=0, num_epoch=num_epoch)
    return model


def train_on_first(args, model, first_path, gt, num_epoch=100):
    metric = mx.metric.CompositeEvalMetric()
    metric.add(extend.MDNetIOUACC())
    metric.add(extend.MDNetIOULoss())
    train_iter = datahelper.get_train_iter(datahelper.get_train_data(first_path, gt))
    model.fit(train_data=train_iter, optimizer='sgd',
              optimizer_params={'learning_rate': args.lr,
                                'wd'           : args.wd,
                                'momentum'     : args.momentum,
                                'clip_gradient': 5,
                                'lr_scheduler' : mx.lr_scheduler.FactorScheduler(
                                    args.lr_step, args.lr_factor, args.lr_stop)},
              eval_metric=metric, begin_epoch=0, num_epoch=num_epoch)
    return model


def track(model, img_path, pre_region, topk=5):
    # only for iou loss
    feat_bboxes = sample.sample_on_feat()
    pred_data, restore_info = datahelper.get_predict_data(img_path, pre_region, feat_bboxes)
    pred_iter = datahelper.get_predict_iter(pred_data)

    def restore_img_bbox(opt_img_bbox, restore_info):
        xo, yo, wo, ho = opt_img_bbox
        img_W, img_H, X, Y, W, H = restore_info
        x, y = W / 227. * xo + X - img_W, H / 227. * yo + Y - img_H
        w, h = W / 227. * wo, H / 227. * ho
        return x, y, w, h

    def check_pred_data(i):
        feat_bbox = feat_bboxes[i, 1:].reshape(1, 4)
        img_bbox = util.feat2img(feat_bbox).reshape(4, )
        img_patch_ = np.reshape(pred_data[0], (227, 227, 3))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img_patch_)
        ax.add_patch(patches.Rectangle((img_bbox[0], img_bbox[1]), img_bbox[2], img_bbox[3],
                                       linewidth=2, edgecolor='y', facecolor='none'))
        fig.show()
        return fig

    # check_pred(0)
    res = model.predict(pred_iter)
    opt_idx = mx.ndarray.topk(res, k=topk).asnumpy().astype('int32')
    opt_scores = res[opt_idx]
    opt_score = opt_scores.mean()
    opt_feat_bboxes = feat_bboxes[opt_idx, 1:]
    opt_patch_bboxes = util.feat2img(opt_feat_bboxes)
    opt_patch_bbox = opt_patch_bboxes.mean(0)
    opt_img_bbox = restore_img_bbox(opt_patch_bbox, restore_info)

    return opt_img_bbox, opt_score


def parse_args():
    parser = argparse.ArgumentParser(description='Train MDNet network')
    parser.add_argument('--gpu', help='GPU device to train with', default=2, type=int)
    parser.add_argument('--num_epoch', help='epoch of training for every frame', default=0, type=int)
    parser.add_argument('--batch_callback_freq', default=50, type=int)
    parser.add_argument('--lr', help='base learning rate', default=1e-5, type=float)
    parser.add_argument('--lr_online', help='base learning rate', default=1e-5, type=float)
    parser.add_argument('--wd', help='base learning rate', default=1e-1, type=float)
    parser.add_argument('--OTB_path', help='OTB folder', default='/home/chenjunjie/dataset/OTB', type=str)
    parser.add_argument('--VOT_path', help='VOT folder', default='/home/chenjunjie/dataset/VOT2015', type=str)
    parser.add_argument('--p_level', help='print level, default is 0 for debug mode', default=0, type=int)
    parser.add_argument('--fixed_conv', help='the params before(include) which conv are all fixed', default=2, type=int)
    parser.add_argument('--loss_type', type=int, default=1,
                        help='0 for {0,1} corss-entropy, 1 for smooth_l1, 2 for {pos_pred} corss-entropy')
    parser.add_argument('--lr_step', default=36 * 1, type=int)
    parser.add_argument('--lr_factor', default=0.9, type=float)
    parser.add_argument('--lr_stop', default=5e-7, type=float)
    parser.add_argument('--iou_acc_th', default=0.1, type=float)
    parser.add_argument('--momentum', default=0, type=float)
    parser.add_argument('--saved_fname', default=None, type=str)
    parser.add_argument('--log', default=1, type=int)

    args = parser.parse_args()
    return args


def track_on_OTB():
    args = parse_args()
    otb = datahelper.OTBHelper(args.OTB_path)


if __name__ == '__main__':
    config.ctx = mx.cpu(0)
    # config.ctx = mx.gpu(2)
    # vot = datahelper.VOTHelper('/home/chenjunjie/dataset/VOT2015')
    vot = datahelper.VOTHelper()
    img_list, gts = vot.get_seq('bag')
    # v0 = datahelper.get_train_iter(datahelper.get_train_data(img_list[0], gts[0]))
    # v1 = datahelper.get_train_iter(datahelper.get_train_data(img_list[1], gts[1]))
    # v2 = datahelper.get_train_iter(datahelper.get_train_data(img_list[2], gts[2]))
    model = extend.init_model(loss_type=1, fixed_conv=0, load_conv123=True, saved_fname='saved/finished_3frame')
    # r0 = model.score(v0, extend.MDNetIOUACC())
    # r1 = model.score(v1, extend.MDNetIOUACC())
    # r2 = model.score(v2, extend.MDNetIOUACC())
    res = track_seq(model, img_list[:10], gts[:10])

    T = 3
    img_path = img_list[T - 1]
    pre_region = gts[T - 2]


    def check_pred_res(res, gt):
        img = plt.imread(img_path)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img)
        ax.add_patch(patches.Rectangle((gt[0], gt[1]), gt[2], gt[3],
                                       linewidth=2, edgecolor='blue', facecolor='none'))
        ax.add_patch(patches.Rectangle((res[0], res[1]), res[2], res[3],
                                       linewidth=2, edgecolor='y', facecolor='none'))
        fig.show()


    res = track(model, img_path, pre_region)
    check_pred_res(res, gts[T - 1])
    a = 1
