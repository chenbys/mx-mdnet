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

try:

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

    model.init_optimizer(kvstore='local', optimizer='sgd',
                         optimizer_params={'learning_rate': args.lr_offline, 'wd': args.wd, 'momentum': args.momentum,
                                           'lr_scheduler': mx.lr_scheduler.FactorScheduler(step=args.lr_step,
                                                                                           factor=args.lr_factor,
                                                                                           stop_factor_lr=args.lr_stop), })

    data_batches = datahelper.get_data_batches(datahelper.get_train_data(img, region))
    logging.info('@CHEN->update %3d.' % len(data_batches))
    for epoch in range(0, args.num_epoch_for_offline):
        t = time()
        for data_batch in data_batches:
            model.forward_backward(data_batch)
            model.update()
        logging.info('| epoch %d, cost:%.4f' % (epoch, time() - t))

    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

    run.add_update_data(img, region)
    regions, probs = [region], [1]

    # Process next
    while True:
        imagefile = handle.frame()
        if not imagefile:
            break
        img = plt.imread(imagefile)
        img_H, img_W, c = np.shape(img)
        cur += 1

        # 初步检测结果
        pre_region = region
        pre_regions = []
        for dx, dy, ws, hs in [[0, 0, 1, 1],
                               [0, 0, 2, 2],
                               [0, 0, 0.5, 0.5]]:
            pre_regions.append(util.central_bbox(pre_region, dx, dy, ws, hs, img_W, img_H))
        pre_regions += util.replace_wh(region, regions[-15:-3:3] + regions[-3:])
        region, prob = run.multi_track(model, img, pre_regions=pre_regions)
        # twice tracking
        if prob > 0.5:
            run.add_update_data(img, region)

            if cur % 10 == 0:
                logging.getLogger().info('@CHEN->long term update')
                model = run.online_update(args, model, 50)
        else:
            logging.getLogger().info('@CHEN->Short term update and Twice tracking')
            model = run.online_update(args, model, 30)
            pre_region = regions[cur - 1]
            # 二次检测时，检查上上次的pre_region，并搜索更大的区域
            pre_regions = util.replace_wh(region, regions[-7:])

            for dx, dy in zip([-0.5, 0, 0.5, 1, 0],
                              [-0.5, 0, 0.5, 0, 1]):
                for ws, hs in zip([0.7, 1, 2],
                                  [0.7, 1, 2]):
                    pre_regions.append(util.central_bbox(pre_region, dx, dy, ws, hs, img_W, img_H))

            region, prob = run.multi_track(model, img, pre_regions=pre_regions)

            if prob > 0.7:
                run.add_update_data(img, region)

        # report result
        logging.getLogger().info('@CHEN->Curf:%d, prob:%5.2f\n' % (cur, prob))
        logging.getLogger().info('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        regions.append(region)
        probs.append(prob)

        handle.report(vot.Rectangle(region[0], region[1], region[2], region[3]))

except Exception as e:
    print '\n=============PRINT EXC=============\n'
    traceback.print_exc()
    print '\n=============FORMAT EXC=============\n'
    print traceback.format_exc()
    print '\n=============CHEN=============\n'
    raise e
    sys.exit(0)
