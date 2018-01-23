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
        logging.getLogger().setLevel(logging.DEBUG)
        if args.loss_type == 0:
            sym = csym.get_mdnet()
        elif args.loss_type == 1:
            # sym = csym.get_mdnet_with_smooth_l1_loss()
            sym = csym.get_mdnet_c()
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

    metric = mx.metric.CompositeEvalMetric()
    if args.loss_type == 0:
        metric.add(extend.MDNetACC())
        metric.add(extend.MDNetLoss())
    else:
        metric.add(extend.MDNetIOUACC(args.iou_acc_th))
        metric.add(extend.MDNetIOUACC(args.iou_acc_th * 2))
        metric.add(extend.MDNetIOUACC(args.iou_acc_th * 3))
        metric.add(extend.MDNetIOULoss())

    p('begin fitting')
    model.fit(train_data=train_iter, eval_data=val_iter, optimizer='sgd',
              optimizer_params={'learning_rate': args.lr,
                                'wd'           : args.wd,
                                'momentum'     : args.momentum,
                                'clip_gradient': 3,
                                'lr_scheduler' : mx.lr_scheduler.FactorScheduler(args.lr_step, args.lr_factor,
                                                                                 args.lr_stop), },
              eval_metric=metric, num_epoch=begin_epoch + num_epoch, begin_epoch=begin_epoch,
              batch_end_callback=mx.callback.Speedometer(1, args.batch_callback_freq))
    p('finish fitting')
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Train MDNet network')
    parser.add_argument('--gpu', help='GPU device to train with', default=2, type=int)
    parser.add_argument('--num_epoch', help='epoch of training', default=30, type=int)
    parser.add_argument('--batch_callback_freq', default=6, type=int)
    parser.add_argument('--lr', help='base learning rate', default=1e-8, type=float)
    parser.add_argument('--wd', help='base learning rate', default=1e-3, type=float)
    parser.add_argument('--OTB_path', help='OTB folder', default='/home/chenjunjie/dataset/OTB', type=str)
    parser.add_argument('--p_level', help='print level, default is 0 for debug mode', default=0, type=int)
    parser.add_argument('--fixed_conv', help='the params before(include) which conv are all fixed', default=2, type=int)
    parser.add_argument('--loss_type', type=int, default=1,
                        help='0 for {0,1} corss-entropy, 1 for smooth_l1, 2 for {pos_pred} corss-entropy')
    parser.add_argument('--lr_step', default=36 * 1, type=int)
    parser.add_argument('--lr_factor', default=0.9, type=float)
    parser.add_argument('--lr_stop', default=1e-10, type=float)
    parser.add_argument('--iou_acc_th', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.5, type=float)

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
        train_iter = datahelper.get_train_iter(datahelper.get_train_data(img_path, gt, iou_label=bool(args.loss_type)))
        val_iter = datahelper.get_train_iter(
            datahelper.get_train_data(img_list[count + 1], gt_list[count + 1], iou_label=bool(args.loss_type)))

        model = train(args, model, train_iter, val_iter, begin_epoch, args.num_epoch,
                      saved_params=saved_params)
        begin_epoch += args.num_epoch
        p('finished training on frame %d.' % count, level=constant.P_RUN)
        count += 1


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


if __name__ == '__main__':
    main()
