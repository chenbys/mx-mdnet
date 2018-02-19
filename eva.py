import mxnet as mx
import numpy as np
import time
from mxnet.base import _as_list
from mxnet.initializer import Uniform

import csym
from setting import config


def predict(arg_params, image_patch, feat_bbox):
    '''
    :param arg_params:
    :param image_patch:
    :param feat_bbox:
    :return: the predict IOU scores for all feat_bbox
    '''

    sym = csym.get_pred_sym()
    data_dic = {'image_patch': image_patch,
                'feat_bbox'  : feat_bbox}
    arg_params.update(data_dic)
    for key, item in arg_params.items():
        arg_params[key] = mx.ndarray.array(item, config.ctx)

    e = sym.bind(config.ctx, arg_params)
    res = e.forward()

    return res[0]


def validate(arg_params, image_patch, feat_bbox, labels):
    '''

    :param arg_params:
    :param image_patch:
    :param feat_bbox:
    :param labels:
    :return: r0,r1,r2 for 0.1acc, 0.2acc, max50 > 0.6, max50 > 0.8
    '''
    scores = predict(arg_params, image_patch, feat_bbox)
    labels_ = labels.reshape((-1,))
    length = scores.shape[0]
    ns = scores.asnumpy()
    nl = labels_.asnumpy()

    # th acc
    subs = ns - nl
    acc0 = np.sum(abs(subs) < 0.1)
    acc1 = np.sum(abs(subs) < 0.2)
    r0 = 1. * acc0 / length
    r1 = 1. * acc1 / length
    # max acc
    # how many pred bbox that top of K will greater than 0.8
    K = 50
    s_idx = mx.ndarray.topk(scores, k=K).asnumpy().astype('int32')
    max_acc0 = np.sum(nl[s_idx] > 0.6)
    max_acc1 = np.sum(nl[s_idx] > 0.8)

    r2 = 1. * max_acc0 / K
    r3 = 1. * max_acc1 / K

    print 'subs < 0.1 : %.2f%% , subs < 0.2 : %.2f%%, max of %d in pred > 0.6 : %.2f%% and > 0.8 : %.2f%%' % (
        r0 * 100, r1 * 100, K, r2 * 100, r3 * 100)
    return r0, r1, r2


def fit(model, train_data, eval_data=None,
        epoch_end_callback=None, kvstore='local',
        optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
        eval_end_callback=None,
        eval_batch_end_callback=None, initializer=Uniform(0.01),
        arg_params=None, aux_params=None, allow_missing=False,
        force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None):
    '''
        validate on train_data
    :return:
    '''
    assert num_epoch is not None, 'please specify number of epochs'
    if eval_data is None:
        eval_data = train_data
    model.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label,
               for_training=True, force_rebind=force_rebind)
    model.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                      allow_missing=allow_missing, force_init=force_init)
    model.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                         optimizer_params=optimizer_params)

    ################################################################################
    # training loop
    ################################################################################
    for epoch in range(begin_epoch, num_epoch):
        tic = time.time()
        nbatch = 0
        data_iter = iter(train_data)
        end_of_batch = False
        next_data_batch = next(data_iter)
        while not end_of_batch:
            data_batch = next_data_batch
            model.forward_backward(data_batch)
            model.update()
            try:
                # pre fetch next batch
                next_data_batch = next(data_iter)
                model.prepare(next_data_batch)
            except StopIteration:
                end_of_batch = True

            nbatch += 1

        # sync aux params across devices
        arg_params, aux_params = model.get_params()
        model.set_params(arg_params, aux_params)

        if epoch_end_callback is not None:
            for callback in _as_list(epoch_end_callback):
                callback(epoch, model.symbol, arg_params, aux_params)

        # ----------------------------------------
        # evaluation on validation set

        # end of 1 epoch, reset the data-iter for another epoch
        train_data.reset()

        # eval on eval_data
        feat_bbox, image_patch, labels = eval_data.data_list
        validate(model.get_params()[0], image_patch, feat_bbox, labels)

        # one epoch of training is finished
        toc = time.time()
        model.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc - tic))
