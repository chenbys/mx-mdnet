import argparse
import logging
import mxnet as mx
import datahelper
import extend
from setting import const
import run
import util
import numpy as np
import time
import easydict
import copy
import matplotlib.pyplot as plt
import os
import debug_run


def train_MD_on_OTB():
    const.check_pre_train = False
    const.check_mc = True

    set_logger()
    args = parse_args()
    logging.getLogger().info(str_args(args))

    otb = datahelper.OTB_VOT_Helper(args.OTB_path)
    seq_names = otb.seq_names
    seq_num = len(seq_names)

    model, all_params = extend.init_model(args)
    sgd = mx.optimizer.SGD(learning_rate=args.lr, wd=args.wd, momentum=args.momentum)
    sgd.set_lr_mult({'fc4_weight': 10, 'fc4_bias': 10,
                     'fc5_weight': 10, 'fc5_bias': 10,
                     'score_bias': 10, 'score_weight': 10})
    model.init_optimizer(kvstore='local', optimizer=sgd, force_init=True)

    begin_k = args.begin_k
    K = seq_num * args.frame_num
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

    ph = datahelper.ParamsHelper()
    ph.load_params(model, begin_k, args.saved_prefix)

    for k in range(begin_k, begin_k + K):
        t = time.time()
        frame_idx, seq_idx = divmod(k, seq_num)
        # frame_idx = 0
        seq_name = seq_names[seq_idx]
        logging.getLogger().info('========================================================')
        logging.getLogger().info('| Seq: %s, k:%6d, frame: %4d' % (seq_name, k, frame_idx))
        img_path, gt = otb.get_data(seq_name, frame_idx)
        img = plt.imread(img_path)
        const.img_H, const.img_W = img.shape[:2]
        if len(img.shape) == 2:
            img = np.stack((img, img, img), 2)

        model = ph.change_params(model, seq_name)
        data_batches = datahelper.get_data_batches(datahelper.get_pre_train_data(img, gt))
        check_metric(model, data_batches)
        for epoch in range(0, args.num_epoch):
            t = time.time()
            model = extend.train_with_hnm(model, data_batches, sel_factor=2)
            # logging.getLogger().info(
            #     '| epoch %d, cost:%.4f, batches: %d ' % (epoch, time.time() - t, len(data_batches)))
            check_metric(model, data_batches)

            end = 1

        end = 1
        if const.check_pre_train == True:
            debug_run.track(model, img, gt, gt, 1, plotc=True)
        ph.update_params(model, seq_name)
        # logging.getLogger('c').info('| time for one iter: %.2f', time.time() - t)
        if k % 300 == 0:
            ph.save_params(model, k, args.saved_prefix)
    return


def check_metric(model, data_batches):
    if not const.check_mc:
        return

    metric = mx.metric.CompositeEvalMetric()
    metric.add([extend.PR(), extend.RR(), extend.TrackTopKACC(), mx.metric.CrossEntropy()])
    for data_batch in data_batches:
        model.forward(data_batch, is_train=False)
        model.update_metric(metric, data_batch.label)

    values = metric.get()[1]
    logging.getLogger().info('|----| Check metric %.2f,%.2f,%.2f, loss:[%.6f]' %
                             (values[0], values[1], values[2], values[3]))


def str_args(args):
    sd = {}
    for key in ['begin_k', 'frame_num', 'wd', 'lr', 'momentum']:
        sd[key] = args.__dict__[key]
    return sd.__str__()


def set_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.basicConfig(
        level=logging.INFO,
        format='| %(message)s. %(asctime)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename='pre_train.log',
        filemode='a')
    # fh = logging.FileHandler('pre_train.log','a')
    # fh.setLevel(logging.DEBUG)
    #
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    #
    formatter = logging.Formatter('| %(message)s. %(asctime)s')
    # fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    #
    # logger.addHandler(fh)
    logger.addHandler(ch)


def parse_args():
    parser = argparse.ArgumentParser(description='Train MDNet network')
    parser.add_argument('--gpu', help='GPU device to train with', default=0, type=int)
    parser.add_argument('--begin_k', default=699, help='continue from this k ', type=int)
    parser.add_argument('--frame_num', default=1000, help='train how many frames for each sequence', type=int)
    parser.add_argument('--saved_prefix', default='k', help='', type=str)

    parser.add_argument('--num_epoch', default=1, help='epoch for each frame training', type=int)

    parser.add_argument('--fixed_conv', default=0, help='these params of [ conv_i <= ? ] will be fixed', type=int)
    parser.add_argument('--OTB_path', help='OTB folder', default='/media/chen/datasets/OTB', type=str)
    parser.add_argument('--VOT_path', help='VOT folder', default='/home/chen/vot-toolkit/cmdnet-workspace/sequences',
                        type=str)

    parser.add_argument('--ROOT_path', help='cmd folder', default='/home/chen/mx-mdnet', type=str)
    parser.add_argument('--wd', default=5e-4, help='weight decay', type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--lr', default=1e-5, help='base learning rate', type=float)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    train_MD_on_OTB()
