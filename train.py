import argparse
import logging
import mxnet as mx
import datahelper
import csym
import extend
from setting import config, constant
from kit import p


def train_on_one_frame(args, img_path, region, model=None, begin_epoch=0, num_epoch=50,
                       val_image_path=None, val_pre_region=None, arg_params=None):
    '''

    :param img_path:
    :param region:
    :return:
    '''
    train_iter = datahelper.get_train_iter(datahelper.get_train_data(img_path, region))
    if val_image_path is not None:
        val_iter = datahelper.get_train_iter(datahelper.get_train_data(val_image_path, val_pre_region))
    else:
        val_iter = None

    if model is None:
        sym = csym.get_mdnet()
        model = mx.mod.Module(symbol=sym, context=config.ctx, data_names=('image_patch', 'feat_bbox',),
                              label_names=('label',),
                              fixed_param_names=('conv1_weight', 'conv1_bias', 'conv2_weight', 'conv2_bias',
                                                 'conv3_weight', 'conv3_bias'))
        model.bind(train_iter.provide_data, train_iter.provide_label)
        if arg_params is not None:
            for k in arg_params.keys():
                arg_params[k] = mx.ndarray.array(arg_params.get(k))
            model.init_params(arg_params=arg_params, allow_missing=True, force_init=False, allow_extra=True)

    logging.getLogger().setLevel(logging.DEBUG)
    metric = mx.metric.CompositeEvalMetric()
    metric.add(extend.MDNetACC())
    metric.add(extend.MDNetLoss())
    p('begin fitting')
    model.fit(train_data=train_iter, eval_data=val_iter, optimizer='sgd',
              optimizer_params={'learning_rate': args.lr,
                                'wd'           : args.wd},
              eval_metric=metric, num_epoch=begin_epoch + num_epoch, begin_epoch=begin_epoch,
              batch_end_callback=mx.callback.Speedometer(1, frequent=10),
              epoch_end_callback=mx.callback.Speedometer(1, frequent=50))
    p('finish fitting')
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Train MDNet network')
    parser.add_argument('--gpu', help='GPU device to train with', default=2, type=int)
    parser.add_argument('--num_epoch', help='epoch of training', default=30, type=int)
    parser.add_argument('--lr', help='base learning rate', default=0.0001, type=float)
    parser.add_argument('--wd', help='base learning rate', default=0.005, type=float)
    parser.add_argument('--OTB_path', help='OTB folder', default='/home/chenjunjie/dataset/OTB', type=str)
    parser.add_argument('--p_level', help='print level, default is 0 for debug mode', default=0, type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config.p_level = args.p_level

    if args.gpu == -1:
        config.ctx = mx.cpu(0)
    else:
        config.ctx = mx.gpu(args.gpu)

    seq_name = 'Surfer'
    otb = datahelper.OTBHelper(args.OTB_path)
    img_list = otb.get_img(seq_name)
    gt_list = otb.get_gt(seq_name)
    model = None
    begin_epoch = 0
    count = 1
    arg_params = extend.get_mdnet_conv123_params()
    for img_path, gt in zip(img_list[0:2], gt_list[0:2]):
        model = train_on_one_frame(args, img_path, gt, model, begin_epoch, args.num_epoch,
                                   val_image_path=img_list[count + 1], val_pre_region=gt_list[count + 1],
                                   arg_params=arg_params)
        begin_epoch += args.num_epoch
        p('finished training on frame %d.' % count, level=constant.P_RUN)
        count += 1


if __name__ == '__main__':
    main()
