import argparse
import logging
import mxnet as mx
import datahelper
import extend
from setting import config, const
from kit import p
import run
import util
import numpy as np
import time


def one_step_train(args, model, train_iter=None, val_iter=None, begin_epoch=0, end_epoch=50):
    metric = mx.metric.CompositeEvalMetric()
    if args.loss_type == 0:
        metric.add(extend.MDNetACC())
        metric.add(extend.MDNetLoss())
    else:
        metric.add(extend.MDNetIOUACC(args.iou_acc_th))
        metric.add(extend.MDNetIOUACC(args.iou_acc_th * 2))
        metric.add(extend.MDNetIOUACC(args.iou_acc_th * 3))
        metric.add(extend.MDNetIOULoss())

    def sf(x):
        # return mx.ndarray.array(x.shape)
        return x

    mon = mx.monitor.Monitor(interval=1, stat_func=sf, pattern='score',
                             sort=True)
    mon = None
    model.fit(train_data=train_iter, eval_data=val_iter, optimizer='sgd',
              optimizer_params={'learning_rate': args.lr,
                                'wd'           : args.wd,
                                'momentum'     : args.momentum,
                                'clip_gradient': 5,
                                'lr_scheduler' : mx.lr_scheduler.FactorScheduler(args.lr_step, args.lr_factor,
                                                                                 args.lr_stop),
                                },
              eval_metric=metric, num_epoch=end_epoch, begin_epoch=begin_epoch,
              batch_end_callback=mx.callback.Speedometer(1, args.batch_callback_freq), monitor=mon)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Train MDNet network')
    parser.add_argument('--gpu', help='GPU device to train with', default=-1, type=int)
    parser.add_argument('--num_epoch', help='epoch of training for every frame', default=1, type=int)
    parser.add_argument('--batch_callback_freq', default=1, type=int)
    parser.add_argument('--lr', help='base learning rate', default=1e-6, type=float)
    parser.add_argument('--wd', help='base learning rate', default=0, type=float)
    parser.add_argument('--OTB_path', help='OTB folder', default='/home/chenjunjie/dataset/OTB', type=str)
    parser.add_argument('--VOT_path', help='VOT folder', default='/home/chenjunjie/dataset/VOT2015', type=str)
    parser.add_argument('--p_level', help='print level, default is 0 for debug mode', default=0, type=int)
    parser.add_argument('--fixed_conv', help='the params before(include) which conv are all fixed', default=0, type=int)
    parser.add_argument('--loss_type', type=int, default=1,
                        help='0 for {0,1} corss-entropy, 1 for smooth_l1, 2 for {pos_pred} corss-entropy')
    parser.add_argument('--lr_step', default=36 * 5, type=int)
    parser.add_argument('--lr_factor', default=0.9, type=float)
    parser.add_argument('--lr_stop', default=1e-7, type=float)
    parser.add_argument('--iou_acc_th', default=0.1, type=float)
    parser.add_argument('--momentum', default=0, type=float)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config.p_level = args.p_level

    if args.gpu == -1:
        config.ctx = mx.cpu(0)
    else:
        config.ctx = mx.gpu(args.gpu)

    vot = datahelper.VOTHelper(args.VOT_path)
    img_list, gt_list = vot.get_seq('bag')
    length = len(img_list)
    model = extend.init_model(args.loss_type, args.fixed_conv, True)
    logging.getLogger().setLevel(logging.DEBUG)
    begin_epoch = 0
    for i in range(length):
        train_iter = datahelper.get_train_iter(
            datahelper.get_train_data(img_list[i], gt_list[i], iou_label=bool(args.loss_type)))
        val_iter = datahelper.get_train_iter(
            datahelper.get_train_data(img_list[(i + 1) % length], gt_list[(i + 1) % length],
                                      iou_label=bool(args.loss_type)))

        model = one_step_train(args, model, train_iter, val_iter, begin_epoch, begin_epoch + args.num_epoch)
        begin_epoch += args.num_epoch
        p('finished training on frame %d.' % i, level=const.P_RUN)


if __name__ == '__main__':
    main()
