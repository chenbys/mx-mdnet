import argparse
import logging
import cv2
import mxnet as mx
import numpy as np
import datahelper
import sample
import symbol
import extend


def train_on_one_frame(args, img_path, region, model=None, begin_epoch=0, num_epoch=50, ctx=mx.gpu(1),
                       val_image_path=None, val_pre_region=None):
    '''

    :param img_path:
    :param region:
    :return:
    '''
    train_iter = datahelper.get_train_iter(img_path, region)
    if val_image_path is not None:
        val_iter = datahelper.get_train_iter(val_image_path, val_pre_region)
    else:
        val_iter = None

    if model is None:
        sym = symbol.get_mdnet()
        model = mx.mod.Module(symbol=sym, context=ctx, data_names=('image_patch', 'feat_bbox',),
                              label_names=('label',))

    logging.getLogger().setLevel(logging.DEBUG)
    model.fit(train_data=train_iter, eval_data=val_iter, optimizer='sgd',
              optimizer_params={'learning_rate': args.lr,
                                'wd'           : args.wd},
              eval_metric=extend.MDNetMetric(), num_epoch=begin_epoch + num_epoch, begin_epoch=begin_epoch,
              batch_end_callback=mx.callback.Speedometer(1))
    print 'finished training on one frame'
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Train MDNet network')
    parser.add_argument('--gpu', help='GPU device to train with', default=1, type=int)
    parser.add_argument('--num_epoch', help='epoch of training', default=1, type=int)
    parser.add_argument('--lr', help='base learning rate', default=0.0001, type=float)
    parser.add_argument('--wd', help='base learning rate', default=0.005, type=float)
    parser.add_argument('--OTB_path', help='OTB folder', default='/Users/chenjunjie/workspace/OTB', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    seq_name = 'Surfer'
    otb = datahelper.OTBHelper(args.OTB_path)
    img_list = otb.get_img(seq_name)[0:30]
    gt_list = otb.get_gt(seq_name)[0:30]
    ctx = mx.gpu(args.gpu)
    model = None
    begin_epoch = 0
    count = 0
    for img_path, gt in zip(img_list, gt_list):
        # validate
        val_img = cv2.imread(img_list[count + 1])
        val_gt = gt_list[count + 1]
        val_iter = datahelper.get_train_iter(val_img, val_gt)

        model = train_on_one_frame(args, img_path, gt, model, begin_epoch, args.num_epoch, ctx, val_iter)
        begin_epoch += args.num_epoch
        count += 1


if __name__ == '__main__':
    main()
