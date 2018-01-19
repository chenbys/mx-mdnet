import argparse
import logging
import mxnet as mx
import datahelper
import csym
import extend
from setting import config, constant
from kit import p


def train(args, model=None, train_iter=None, val_iter=None, begin_epoch=0, num_epoch=50, saved_params=None):
    '''

    :param img_path:
    :param region:
    :return:
    '''
    if model is None:
        sym = csym.get_mdnet()
        fixed_param_names = []
        for i in range(1, args.fixed_conv + 1):
            fixed_param_names.append('conv' + str(i) + '_weight')
            fixed_param_names.append('conv' + str(i) + '_bias')
        model = mx.mod.Module(symbol=sym, context=config.ctx, data_names=('image_patch', 'feat_bbox',),
                              label_names=('label',),
                              fixed_param_names=fixed_param_names)
        model.bind(train_iter.provide_data, train_iter.provide_label)
        if saved_params is not None:
            for k in saved_params.keys():
                saved_params[k] = mx.ndarray.array(saved_params.get(k))
            model.init_params(arg_params=saved_params, allow_missing=True, force_init=False, allow_extra=True)

    logging.getLogger().setLevel(logging.DEBUG)
    metric = mx.metric.CompositeEvalMetric()
    metric.add(extend.MDNetACC())
    metric.add(extend.MDNetLoss())
    p('begin fitting')
    model.fit(train_data=train_iter, eval_data=val_iter, optimizer='sgd',
              optimizer_params={'learning_rate': args.lr,
                                'wd'           : args.wd},
              eval_metric=metric, num_epoch=begin_epoch + num_epoch, begin_epoch=begin_epoch)
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
    parser.add_argument('--fixed_conv', help='the params before(include) which conv are all fixed', default=2, type=int)

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
    saved_params = extend.get_mdnet_conv123_params()
    for img_path, gt in zip(img_list[0:2], gt_list[0:2]):
        train_iter = datahelper.get_train_iter(datahelper.get_train_data(img_path, gt))
        val_iter = datahelper.get_train_iter(datahelper.get_train_data(img_list[count + 1], gt_list[count + 1]))

        model = train(args, model, train_iter, val_iter, begin_epoch, args.num_epoch,
                      saved_params=saved_params)
        begin_epoch += args.num_epoch
        p('finished training on frame %d.' % count, level=constant.P_RUN)
        count += 1


if __name__ == '__main__':
    main()


def test_track_speed():
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
    img_path = img_list[0]
    gt = gt_list[0]
    import track
    import time
    train_iter = datahelper.get_train_iter(datahelper.get_train_data(img_path, gt))
    model = train(args, None, train_iter, num_epoch=1)
    t1 = time.time()
    box = track.track(model, img_path, gt)
    t2 = time.time()
    print t2 - t1
    print box
