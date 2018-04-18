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
    model = extend.init_model(args)
    logging.getLogger().setLevel(logging.INFO)
    const.img_H, const.img_W, c = img.shape

    sgd = mx.optimizer.SGD(learning_rate=args.lr_offline, wd=args.wd, momentum=args.momentum)
    sgd.set_lr_mult({'fc4_bias': 2, 'fc5_bias': 2, 'score_bias': 20, 'score_weight': 10})
    model.init_optimizer(kvstore='local', optimizer=sgd, force_init=True)

    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    run.offline_update(args, model, img, region)

    sgd = mx.optimizer.SGD(learning_rate=args.lr_online, wd=args.wd, momentum=args.momentum)
    sgd.set_lr_mult({'fc4_bias': 2, 'fc5_bias': 2, 'score_bias': 20, 'score_weight': 10})
    model.init_optimizer(kvstore='local', optimizer=sgd, force_init=True)
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

    run.add_update_data(img, region)
    last_update = -5
    regions, probs = [region], [0.8]
    bh = util.BboxHelper(region)

    # Process next
    while True:
        imagefile = handle.frame()
        if not imagefile:
            break
        img = plt.imread(imagefile)
        cur += 1

        # 初步检测结果
        pre_regions = bh.get_base_regions()
        B, P = run.multi_track(model, img, pre_regions=pre_regions)
        region, prob = util.refine_bbox(B, P, regions[-1])
        # twice tracking
        if (prob > 0.5) & (prob > (probs[-1] - 0.1)):

            run.add_update_data(img, region, B)

            if cur - last_update > 10:
                logging.info('| long term update')
                model = run.online_update(args, model, 50, const.update_batch_num / 8)
                last_update = cur

        else:
            logging.info('| twice tracking %d.jpg for prob: %.6f' % (cur, prob))

            if cur - last_update > 1:
                logging.info('| short term update')
                model = run.online_update(args, model, 5, const.update_batch_num)
                last_update = cur

            pre_regions = bh.get_twice_base_regions()
            B, P = run.multi_track(model, img, pre_regions=pre_regions)
            region, prob = util.refine_bbox(B, P, regions[-1])

            if prob < 0.8:
                region = regions[-1]
            else:
                run.add_update_data(img, region, B)

        # report result
        bh.add_res(region)

        logging.getLogger().info('@CHEN->Curf:%d, prob:%5.2f\n' % (cur, prob))
        logging.getLogger().info('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

        regions.append(region)
        probs.append(prob)

        handle.report(vot.Rectangle(region[0], region[1], max(15, region[2]), max(15, region[3])))

except Exception as e:
    print '\n=============PRINT EXC=============\n'
    traceback.print_exc()
    print '\n=============FORMAT EXC=============\n'
    print traceback.format_exc()
    print '\n=============CHEN=============\n'
    raise e
    sys.exit(0)
