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
    train_iter = datahelper.get_iter(datahelper.get_train_data(img, region))
    model.fit(train_data=train_iter, optimizer='sgd',
              eval_metric=mx.metric.CompositeEvalMetric(
                  [extend.SMLoss(), extend.PR(0.5), extend.RR(0.5), extend.TrackTopKACC(10, 0.6)]),
              optimizer_params={'learning_rate': args.lr_offline, 'wd': args.wd, 'momentum': args.momentum},
              begin_epoch=0, num_epoch=args.num_epoch_for_offline)

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
                               [-0.5, -0.5, 2, 2],
                               [0, 0, 1.5, 1.5],
                               [0, 0, 0.5, 0.5]]:
            pre_regions.append(util.central_bbox(pre_region, dx, dy, ws, hs, img_W, img_H))

        region, prob = run.multi_track(model, img, pre_regions=pre_regions)
        # twice tracking
        if prob > 0.5:
            run.add_update_data(img, region)

            if cur % 10 == 0:
                logging.getLogger().info('@CHEN->long term update')
                model = run.online_update(args, model, 100)
        else:
            logging.getLogger().info('@CHEN->Short term update and Twice tracking')
            model = run.online_update(args, model, 30)
            pre_region = regions[cur - 1]
            # 二次检测时，检查上上次的pre_region，并搜索更大的区域
            pre_regions = [regions[max(0, cur - 2)]]
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for ws in [0.5, 1, 2]:
                        for hs in [0.5, 1, 2]:
                            pre_regions.append(util.central_bbox(pre_region, dx, dy, ws, hs, img_W, img_H))

            region, prob = run.multi_track(model, img, pre_regions=pre_regions)

            if prob > 0.5:
                run.add_update_data(img, region)

        # report result
        logging.getLogger().info('\n')
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
