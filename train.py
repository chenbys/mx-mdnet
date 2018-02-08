import argparse
import logging
import mxnet as mx
import datahelper
import extend
from setting import config, constant
from kit import p
import run
import util
import numpy as np
import time
import easydict
import copy


def train_MD_on_VOT():
    args = parse_args()
    config.p_level = args.p_level
    if args.gpu == -1:
        config.ctx = mx.cpu(0)
    else:
        config.ctx = mx.gpu(args.gpu)
    model = extend.init_model(loss_type=args.loss_type, fixed_conv=args.fixed_conv, saved_fname=args.saved_fname)

    vot = datahelper.VOTHelper(args.VOT_path)
    if args.log == 0:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.ERROR)
    all_params = {}

    # seq num
    N = 60
    # training step num for each seq , one step for one frame
    K = N * 5 * 1000

    for k in range(K):
        seq_idx = k % 60
        seq_name = vot.seq_names[seq_idx]

        # load params for MD training
        arg_params, aux_params = model.get_params()
        arg_params = extend.get_params(seq_name, arg_params, all_params)
        model.set_params(arg_params=arg_params, aux_params=aux_params,
                         allow_missing=True, force_init=True, allow_extra=False)

        # train
        rep_times, train_iter, val_iter = vot.get_data(k)
        print '@CHEN->%d iter, in %s, %d repeat times' % (k, seq_name, rep_times)
        model = one_step_train(args, model, train_iter, end_epoch=args.num_epoch)

        # validate
        train_res = model.score(train_iter, extend.MDNetIOUACC())
        val_res = model.score(val_iter, extend.MDNetIOUACC())
        for name, val in train_res:
            logging.getLogger().error('@CHEN->train-%s=%f', name, val)
        for name, val in val_res:
            logging.getLogger().error('@CHEN->valid-%s=%f', name, val)

        # save params for MD training
        arg_params, aux_params = model.get_params()
        all_params[seq_name] = copy.deepcopy(arg_params)


def train_SD_on_VOT():
    args = parse_args()
    config.p_level = args.p_level

    if args.gpu == -1:
        config.ctx = mx.cpu(0)
    else:
        config.ctx = mx.gpu(args.gpu)
    model = extend.init_model(loss_type=args.loss_type, fixed_conv=args.fixed_conv, saved_fname=args.saved_fname)

    vot = datahelper.VOTHelper(args.VOT_path)
    if args.log == 0:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.ERROR)

    N = 5
    for n in range(N):
        for seq_name in vot.seq_names:
            print '@CHEN->%s in %d/%d circ' % (seq_name, n, N)
            img_list, gt_list = vot.get_seq(seq_name)
            length = len(img_list)
            for i in range(length):
                begin_epoch = 0 + n * 10
                print '@CHEN->frame:%d/%d' % (i, length)
                train_iter = datahelper.get_train_iter(
                    datahelper.get_train_data(img_list[i], gt_list[i], iou_label=bool(args.loss_type)))
                val_iter = datahelper.get_train_iter(
                    datahelper.get_train_data(img_list[(i + 1) % length], gt_list[(i + 1) % length],
                                              iou_label=bool(args.loss_type)))
                model = one_step_train(args, model, train_iter, val_iter, begin_epoch, begin_epoch + args.num_epoch)
                begin_epoch += args.num_epoch

                # val
                train_res = model.score(train_iter, extend.MDNetIOUACC())
                val_res = model.score(val_iter, extend.MDNetIOUACC())
                for name, val in train_res:
                    logging.getLogger().error('train-%s=%f', name, val)
                for name, val in val_res:
                    logging.getLogger().error('valid-%s=%f', name, val)

                    # try tracking for validation
                    # res = run.track(model, img_list[(i + 1) % length], gt_list[i])
                    # print '@CHEN->track on frame %d, iou of res is %.2f' % (
                    #     i + 1, util.overlap_ratio(res, np.array(gt_list[(i + 1) % length])))
                    # print res
                    # print gt_list[(i + 1) % length]
                    # res2 = run.track(model, img_list[(i) % length], gt_list[(i - 1) % length])
                    # print '@CHEN->track on frame %d, iou of res is %.2f' % (
                    #     i, util.overlap_ratio(res2, np.array(gt_list[(i) % length])))
                    # print res2
                    # print gt_list[(i) % length]
            # save and load
            model.save_params('saved/%s_%dcirc' % (seq_name, n))


def one_step_train(args, model, train_iter=None, val_iter=None, begin_epoch=0, end_epoch=50):
    '''

    :param img_path:
    :param region:
    :return:
    '''
    metric = mx.metric.CompositeEvalMetric()
    metric.add(extend.MDNetIOUACC(args.iou_acc_th))
    metric.add(extend.MDNetIOUACC(args.iou_acc_th * 2))
    metric.add(extend.MDNetIOUACC(args.iou_acc_th * 3))
    metric.add(extend.MDNetIOULoss())

    t1 = time.time()
    model.fit(train_data=train_iter, eval_data=val_iter, optimizer='sgd',
              optimizer_params={'learning_rate': args.lr,
                                'wd'           : args.wd,
                                'momentum'     : args.momentum,
                                'clip_gradient': 5,
                                'lr_scheduler' : mx.lr_scheduler.FactorScheduler(
                                    args.lr_step, args.lr_factor, args.lr_stop)},
              eval_metric=metric, num_epoch=end_epoch, begin_epoch=begin_epoch,
              batch_end_callback=mx.callback.Speedometer(1, args.batch_callback_freq))
    t2 = time.time()

    print '@CHEN->one_step cost %.2f' % (t2 - t1)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Train MDNet network')
    parser.add_argument('--gpu', help='GPU device to train with', default=2, type=int)
    parser.add_argument('--num_epoch', help='epoch of training for every frame', default=0, type=int)
    parser.add_argument('--batch_callback_freq', default=50, type=int)
    parser.add_argument('--lr', help='base learning rate', default=1e-6, type=float)
    parser.add_argument('--wd', help='base learning rate', default=0, type=float)
    parser.add_argument('--OTB_path', help='OTB folder', default='/home/chenjunjie/dataset/OTB', type=str)
    parser.add_argument('--VOT_path', help='VOT folder', default='/home/chenjunjie/dataset/VOT2015', type=str)
    parser.add_argument('--p_level', help='print level, default is 0 for debug mode', default=0, type=int)
    parser.add_argument('--fixed_conv', help='the params before(include) which conv are all fixed', default=0, type=int)
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


if __name__ == '__main__':
    train_MD_on_VOT()
