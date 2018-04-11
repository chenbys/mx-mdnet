# -*-coding:utf- 8-*-

import logging
import sys
from time import time
import matplotlib.pyplot as plt
import numpy as np
import util
import vot
import mxnet as mx
import extend
import datahelper
import run
import traceback
import os

from setting import const

try:
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '1'
    handle = vot.VOT("rectangle")
    region = handle.region()

    # Process the first frame
    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)
    img = plt.imread(imagefile)

    cur = 0

    args = run.parse_args()
    model, all_params = extend.init_model(args)
    logging.getLogger().setLevel(logging.INFO)

    const.img_H, const.img_W, c = img.shape
    sgd = mx.optimizer.SGD(learning_rate=args.lr_offline, wd=args.wd, momentum=args.momentum)
    sgd.set_lr_mult({'score_bias': 10, 'score_weight': 10})
    model.init_optimizer(kvstore='local', optimizer=sgd, force_init=True)

    data_batches = datahelper.get_data_batches(datahelper.get_train_data(img, region))
    logging.info('@CHEN->update %3d.' % len(data_batches))
    batch_mc = extend.ACC()
    for epoch in range(0, args.num_epoch_for_offline):
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

            # 找出高loss的多更新
            sorted_idx = np.argsort(batch_mc_log)
            sel = len(batch_idx) * 4 / 5
            if sel < len(data_batches) / 10:
                break
            batch_idx = batch_idx[sorted_idx[-sel:]]

    sgd = mx.optimizer.SGD(learning_rate=args.lr_online, wd=args.wd, momentum=args.momentum)
    sgd.set_lr_mult({'score_bias': 10, 'score_weight': 10})
    model.init_optimizer(kvstore='local', optimizer=sgd, force_init=True)

    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

    run.add_update_data(img, region, 0)
    regions, probs = [region], [0.8]
    last_update = -5
    bh = util.BboxHelper(region)

    # Process next
    while True:
        imagefile = handle.frame()
        if not imagefile:
            break
        img = plt.imread(imagefile)
        # img_H, img_W, c = np.shape(img)
        cur += 1

        # 初步检测结果
        pre_regions = bh.get_base_regions()
        B, P = run.multi_track(model, img, pre_regions=pre_regions)
        region, prob = util.refine_bbox(B, P, regions[-1])
        # twice tracking
        if (prob > 0.5) & (prob > (probs[-1] - 0.1)):

            run.add_update_data(img, region, cur)

            if cur - last_update > 10:
                logging.info('| long term update')
                model = run.online_update(args, model, 20, 2)
                last_update = cur

        else:
            logging.info('| short term update for porb: %.2f' % prob)
            if cur - last_update > 5:
                model = run.online_update(args, model, 5, 1)
                last_update = cur
            logging.info('| twice tracking %d.jpg' % cur)

            pre_regions = bh.get_twice_base_regions()
            B, P = run.multi_track(model, img, pre_regions=pre_regions)
            region, prob = util.refine_bbox(B, P, regions[-1])

            if prob < 0.5:
                region = regions[-1]

        # report result
        bh.add_res(region)

        logging.getLogger().info('@CHEN->Curf:%d, prob:%5.2f\n' % (cur, prob))
        logging.getLogger().info('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

        regions.append(region)
        probs.append(prob)

        handle.report(vot.Rectangle(region[0], region[1], max(15, region[2]), max(15, region[3]) ))

except Exception as e:
    print '\n=============PRINT EXC=============\n'
    traceback.print_exc()
    print '\n=============FORMAT EXC=============\n'
    print traceback.format_exc()
    print '\n=============CHEN=============\n'
    raise e
    sys.exit(0)
