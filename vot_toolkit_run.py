# -*-coding:utf- 8-*-

import logging
import sys
from time import time
import matplotlib.pyplot as plt
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
                  [extend.PR(0.5), extend.RR(0.5), extend.TrackTopKACC(10, 0.6)]),
              optimizer_params={'learning_rate': args.lr_offline,
                                'wd': args.wd,
                                'momentum': args.momentum},
              begin_epoch=0, num_epoch=args.num_epoch_for_offline)

    run.add_update_data(img, region)
    regions, probs = [region], [1]

    # Process next
    while True:
        imagefile = handle.frame()
        if not imagefile:
            break
        TT = time()
        img = plt.imread(imagefile)
        logging.getLogger().info('@CHEN-> time for imread:%.6f' % (time() - TT))

        cur += 1

        # 初步检测结果
        T1 = time()
        region, prob = run.track(model, img, pre_region=region)
        T2 = time()

        # prepare online update data
        if prob > 0.6:
            T3 = time()
            run.add_update_data(img, region)
            T4 = time()
            logging.getLogger().info(
                '@CHEN->Time for track:%8.6f,Time for add_update_data:%8.6f\n' % (T2 - T1, T4 - T3))

        # online update
        if prob < 0.6:
            # short term update
            logging.getLogger().info('@CHEN->Short term update')
            model = run.online_update(args, model, 20)
            # cur = cur - 1
        elif cur % 10 == 0:
            # long term update
            logging.getLogger().info('@CHEN->Long term update')
            model = run.online_update(args, model, 100)

        # 汇报检测结果
        logging.getLogger().info('\n')
        logging.getLogger().info('@CHEN->Curf:%d, prob:%5.2f\n' % (cur, prob))
        logging.getLogger().info('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        regions.append(region)
        probs.append(prob)

        handle.report(vot.Rectangle(region[0], region[1], region[2], region[3]))

except Exception as e:
    traceback.print_exc()
    print '\n=============CHEN=============\n'
    print traceback.format_exc()
    print '\n=============CHEN=============\n'
    raise e
