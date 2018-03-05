import logging
import mxnet as mx
import numpy as np
import time

from matplotlib import patches
from mxnet.initializer import Uniform
import copy
import csym
import datahelper
import sample
import util
from setting import config
import matplotlib.pyplot as plt


def predict(arg_params, image_patch, feat_bbox):
    '''
        This func will modify arg_params, so pass one copy
    :param arg_params:
    :param image_patch:
    :param feat_bbox:
    :return: the predict IOU scores for all feat_bbox
    '''
    sym = csym.get_pred_sym()
    data_dic = {'image_patch': image_patch,
                'feat_bbox': feat_bbox}
    arg_params.update(data_dic)
    for key, item in arg_params.items():
        arg_params[key] = mx.ndarray.array(item, config.ctx)

    e = sym.bind(config.ctx, arg_params)
    res = e.forward()

    return res[0]


def val_for_tracking(arg_params, img_path, pre_region, gt):
    '''
        This func will modify arg_params, so pass one copy

    :param arg_params:
    :param img_path:
    :param pre_region:
    :param gt:
    :return:
    '''

    import run
    region, score = run.track(arg_params, img_path, pre_region)
    iou = util.overlap_ratio(region, gt)
    d = util.overlap_ratio(pre_region, gt)
    logging.getLogger().info('track on %s, res iou is %5.2f, base iou  is %5.2f' % (img_path, iou, d))
    return iou


def val_for_overfitting(arg_params, train_img_path, train_gt):
    train_iter = datahelper.get_train_iter(datahelper.get_train_data(train_img_path, train_gt))
    val_for_fitting(arg_params, train_iter)


def val_for_fitting(arg_params, train_iter):
    '''
        This func will modify arg_params, so pass one copy

    :param arg_params:
    :param image_patch:
    :param feat_bbox:
    :param labels:
    :return: r0,r1,r2 for 0.1acc, 0.2acc, max50 > 0.6, max50 > 0.8
    '''
    feat_bbox, image_patch, labels = train_iter.data_list
    scores_ = predict(arg_params, image_patch, feat_bbox)

    # reshape scores like as labels
    patch_num, label_num = labels.shape
    scores = scores_.reshape((patch_num, label_num))

    nscores = scores.asnumpy()
    nlabels = labels.asnumpy()

    def check_fit_plot(patch_idx):
        plt.plot(nlabels[patch_idx, :])
        plt.plot(nscores[patch_idx, :])

    def check_fit(patch_idx, sample_idx):
        img_patches, feat_bboxes = image_patch.asnumpy(), feat_bbox.asnumpy()
        f_bbox = feat_bboxes[patch_idx, sample_idx, :]
        label = nlabels[patch_idx, sample_idx]
        score = nscores[patch_idx, sample_idx]
        img_patch = img_patches[patch_idx, :, :, :]

        patch_bbox = util.feat2img(f_bbox[1:].reshape((1, 4))).reshape(4, )
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img_patch.reshape((227, 227, 3)))
        ax.add_patch(patches.Rectangle((patch_bbox[0], patch_bbox[1]), patch_bbox[2], patch_bbox[3],
                                       linewidth=2, edgecolor='red', facecolor='none'))
        fig.show()
        return label, score

    # val_for_tracking(arg_params, config.img_paths[31], config.gts[30], config.gts[31])
    topK = 10
    r0s, r1s, r2s, r3s = [], [], [], []
    for patch_idx in range(0, patch_num, 1):
        score = nscores[patch_idx, :]
        label = nlabels[patch_idx, :]
        r0 = get_subs_acc(score, label, 0.1)
        r1 = get_subs_acc(score, label, 0.2)
        r2 = get_topK_acc(score, label, topK, 0.6)
        r3 = get_topK_acc(score, label, topK, 0.8)
        r0s.append(r0)
        r1s.append(r1)
        r2s.append(r2)
        r3s.append(r3)

        logging.getLogger().info('pid:%3d|%6.0f%%|%6.0f%%|%6.0f%%|%6.0f%%' % (
            patch_idx, r0 * 100, r1 * 100, r2 * 100, r3 * 100))

    logging.getLogger().info('mean of above: %6.2f|%6.2f|%6.2f|%6.2f|' % (
        np.mean(r0s), np.mean(r1s), np.mean(r2s), np.mean(r3s)))
    return


def get_subs_acc(score, label, subs_th):
    subs = score - label
    acc = np.sum(abs(subs) < subs_th)
    r = 1. * acc / len(score)
    return r


def get_topK_acc(score, label, topK, th):
    topK_idx = mx.ndarray.topk(mx.ndarray.array(score), k=topK).asnumpy().astype('int32')
    topK_acc = np.sum(label[topK_idx] > th)
    r = 1. * topK_acc / topK
    return r


def fit(model, train_img_path, train_gt, val_img_path, val_pre_region, val_gt,
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
    train_iter = datahelper.get_train_iter(datahelper.get_train_data(train_img_path, train_gt))

    model.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label,
               for_training=True, force_rebind=force_rebind)
    model.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                      allow_missing=allow_missing, force_init=force_init)
    model.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                         optimizer_params=optimizer_params)

    ################################################################################
    # training loop
    ################################################################################
    logging.getLogger().info('=================== INIT ====================')
    val_for_fitting(copy.deepcopy(model.get_params()[0]), train_iter)
    val_for_tracking(copy.deepcopy(model.get_params()[0]), val_img_path, val_pre_region, val_gt)

    for epoch in range(begin_epoch, num_epoch):
        tic = time.time()
        nbatch = 0
        data_iter = iter(train_iter)
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
        # arg_params, aux_params = model.get_params()
        # model.set_params(arg_params, aux_params)

        # if epoch_end_callback is not None:
        #     for callback in _as_list(epoch_end_callback):
        #         callback(epoch, model.symbol, arg_params, aux_params)

        # ----------------------------------------
        # evaluation on validation set

        # end of 1 epoch, reset the data-iter for another epoch
        train_iter.reset()

        # eval on eval_data
        if (epoch + 1) % 50 == 0:
            logging.getLogger().info('\n============= Epoch:%4d .====================' % (epoch + 1))
            val_for_fitting(copy.deepcopy(model.get_params()[0]), train_iter)
            # val_for_overfitting(copy.deepcopy(model.get_params()[0]), train_img_path, train_gt)
            val_for_tracking(copy.deepcopy(model.get_params()[0]), val_img_path, val_pre_region, val_gt)

        # one epoch of training is finished
        toc = time.time()
        # model.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc - tic))
    return
