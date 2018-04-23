# -*-coding:utf- 8-*-
from collections import OrderedDict
from time import time

import logging
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
import csym
import kit
import util
from setting import const
import copy
import os


def train_with_hnm(model, data_batches, sel_factor=3):
    hard_batches = data_batches
    while True:
        temp_batches = []
        t = time()
        for data_batch in hard_batches:
            # model.forward_backward(data_batch)
            # model.update()
            model.forward(data_batch, is_train=False)

            # 250,
            label = data_batch.label[0][0].asnumpy()
            # shape: 250,2
            pred = model._exec_group.execs[0].outputs[0]
            pos_prob = pred.asnumpy()[:, 1]

            # 正样本的idx
            pos_samples_idx_ = np.argwhere(label == 1).reshape((-1))
            pos_samples_idx = pos_samples_idx_[np.argwhere(pos_prob[pos_samples_idx_] < 0.8).reshape((-1))]
            pos_num = pos_samples_idx.shape[0] / sel_factor
            # 正样本的输出分值
            pos_samples_prob = pos_prob[pos_samples_idx]
            # 正样本分值排序
            pos_sel_idx = pos_samples_idx[np.argsort(pos_samples_prob)[:pos_num]]

            # mining idx for hard samples
            ## topK输出分值的负样本
            ### 负样本的idx
            neg_samples_idx_ = np.argwhere(label == 0).reshape((-1))
            neg_samples_idx = neg_samples_idx_[np.argwhere(pos_prob[neg_samples_idx_] > 0.1).reshape((-1))]
            neg_num = neg_samples_idx.shape[0] / sel_factor

            ### 负样本的输出分值
            neg_samples_prob = pos_prob[neg_samples_idx]
            ### 负样本分值排序
            neg_sel_idx = neg_samples_idx[np.argsort(neg_samples_prob)[-neg_num:]]

            # 选出的样本的idx
            sel_idx = np.hstack((pos_sel_idx, neg_sel_idx))
            if len(sel_idx) <= 10:
                continue

            # 1,3,329,324
            img_patch = data_batch.data[1]

            # 1,250,5
            feat_bbox = data_batch.data[0]
            hard_batch = mx.io.DataBatch([copy.deepcopy(feat_bbox[:, sel_idx, :]), copy.deepcopy(img_patch)],
                                         [mx.ndarray.array([copy.deepcopy(label[sel_idx])])])
            temp_batches.append(hard_batch)

            def check_sample(i=neg_sel_idx):
                ig = np.transpose(img_patch[0].asnumpy(), const.NHW2HWN)
                fb = feat_bbox[0, i, :].asnumpy()
                ib = util.feat2img(fb[:, 1:])
                kit.show_tracking(ig, ib)
                return pos_prob[i]

            model.forward_backward(hard_batch)
            model.update()

            a = 1

        logging.info('| cost %.6f, batches %d->%d' % (time() - t, len(hard_batches), len(temp_batches)))
        if len(temp_batches) < len(data_batches) / 4:
            break
        hard_batches = temp_batches

    return model


class PR(mx.metric.EvalMetric):
    def __init__(self, pos_th=0.7):
        '''
            评价模型的准确率：判断准的正样本数量/判断是正的样本数量
        :param pos_th: iou >= pos_th 认为是正样本
        '''
        super(PR, self).__init__('PR')
        self.pos_th = pos_th

    def update(self, labels, preds):
        '''
        :param labels:
        :param preds:
        :return:
        '''
        labels = labels[0].asnumpy()[0, :]
        scores = preds[0].asnumpy()
        output_pos_scores = scores[:, 1]
        output_pos_idx = output_pos_scores >= self.pos_th
        hit = np.sum(labels[output_pos_idx] > self.pos_th)
        length = np.sum(output_pos_idx)

        self.sum_metric += hit
        self.num_inst += length


class RR(mx.metric.EvalMetric):
    def __init__(self, pos_th=0.7):
        '''
            评价模型的准确率：判断准的正样本数量/所有正样本数量
        :param pos_th: iou >= pos_th 认为是正样本
        '''
        super(RR, self).__init__('RR')
        self.pos_th = pos_th

    def update(self, labels, preds):
        '''
        :param labels:
        :param preds:
        :return:
        '''
        labels = labels[0].asnumpy()[0, :]
        scores = preds[0].asnumpy()
        output_pos_scores = scores[:, 1]
        output_pos_idx = output_pos_scores >= self.pos_th
        hit = np.sum(labels[output_pos_idx] > self.pos_th)
        length = np.sum(labels > self.pos_th)

        self.sum_metric += hit
        self.num_inst += max(1, length)


class TrackTopKACC(mx.metric.EvalMetric):
    def __init__(self, topK=5, th=0.7):
        '''
            评价模型输出概率的最大K个样本对应label大于th的比率
        :param topK:
        '''
        super(TrackTopKACC, self).__init__('TrackTopKAcc')
        self.topK = topK
        self.th = th

    def update(self, labels, preds):
        labels = labels[0].asnumpy()[0, :]
        scores = preds[0].asnumpy()
        pos_scores = scores[:, 1]
        self.topK = min(self.topK, pos_scores.shape[0])
        topK_idx = pos_scores.argsort()[-self.topK::]
        # 只考虑topK中的被认为是正样本的idx
        topK_idx = topK_idx[pos_scores[topK_idx] > self.th]
        topK_acc = np.sum(labels[topK_idx] > self.th)
        self.sum_metric += topK_acc
        self.num_inst += max(1e-5, topK_idx.shape[0])


class ACC(mx.metric.EvalMetric):
    def __init__(self):
        super(ACC, self).__init__('ACC')

    def update(self, labels, preds):
        labels = labels[0].as_in_context(mx.gpu(0))
        scores = preds[0]
        hit = mx.ndarray.sum(mx.ndarray.argmax(scores, 1) == labels.reshape((-1,)))
        self.sum_metric += hit.asnumpy()
        self.num_inst += labels.shape[1]


def get_mdnet_conv123_params(mat_path, prefix=''):
    import scipy.io as sio
    import numpy as np
    conv123 = sio.loadmat(mat_path)
    conv123 = conv123['conv123']
    conv1_weight = conv123[0, 0][0]
    conv1_bias = conv123[0, 0][1]
    conv2_weight = conv123[0, 0][2]
    conv2_bias = conv123[0, 0][3]
    conv3_weight = conv123[0, 0][4]
    conv3_bias = conv123[0, 0][5]

    conv1_weight_ = np.transpose(conv1_weight, [3, 2, 0, 1])
    conv1_bias_ = conv1_bias.reshape((96,))
    conv2_weight_ = np.transpose(conv2_weight, [3, 2, 0, 1])
    conv2_bias_ = conv2_bias.reshape((256,))
    conv3_weight_ = np.transpose(conv3_weight, [3, 2, 0, 1])
    conv3_bias_ = conv3_bias.reshape((512,))

    arg_params = dict()
    arg_params[prefix + 'conv1_weight'] = conv1_weight_
    arg_params[prefix + 'conv1_bias'] = conv1_bias_
    arg_params[prefix + 'conv2_weight'] = conv2_weight_
    arg_params[prefix + 'conv2_bias'] = conv2_bias_
    arg_params[prefix + 'conv3_weight'] = conv3_weight_
    arg_params[prefix + 'conv3_bias'] = conv3_bias_

    return arg_params


def get_mdnet_conv123fc4fc5_params(mat_path, prefix=''):
    import scipy.io as sio
    import numpy as np
    conv123 = sio.loadmat(mat_path)
    conv123 = conv123['conv123']
    conv1_weight = conv123[0, 0][0]
    conv1_bias = conv123[0, 0][1]
    conv2_weight = conv123[0, 0][2]
    conv2_bias = conv123[0, 0][3]
    conv3_weight = conv123[0, 0][4]
    conv3_bias = conv123[0, 0][5]
    fc4_weight = conv123[0, 0][6]
    fc4_bias = conv123[0, 0][7]
    fc5_weight = conv123[0, 0][8]
    fc5_bias = conv123[0, 0][9]

    conv1_weight_ = np.transpose(conv1_weight, [3, 2, 0, 1])
    conv1_bias_ = conv1_bias.reshape((96,))
    conv2_weight_ = np.transpose(conv2_weight, [3, 2, 0, 1])
    conv2_bias_ = conv2_bias.reshape((256,))
    conv3_weight_ = np.transpose(conv3_weight, [3, 2, 0, 1])
    conv3_bias_ = conv3_bias.reshape((512,))
    fc4_weight_ = np.transpose(fc4_weight, [3, 2, 0, 1])
    fc4_bias_ = fc4_bias.reshape((512,))
    fc5_weight_ = np.transpose(fc5_weight, [3, 2, 0, 1])
    fc5_bias_ = fc5_bias.reshape((512,))

    arg_params = dict()
    arg_params[prefix + 'conv1_weight'] = conv1_weight_
    arg_params[prefix + 'conv1_bias'] = conv1_bias_
    arg_params[prefix + 'conv2_weight'] = conv2_weight_
    arg_params[prefix + 'conv2_bias'] = conv2_bias_
    arg_params[prefix + 'conv3_weight'] = conv3_weight_
    arg_params[prefix + 'conv3_bias'] = conv3_bias_
    arg_params[prefix + 'fc4_weight'] = fc4_weight_
    arg_params[prefix + 'fc4_bias'] = fc4_bias_
    arg_params[prefix + 'fc5_weight'] = fc5_weight_
    arg_params[prefix + 'fc5_bias'] = fc5_bias_

    return arg_params


def get_mdnet_conv123fc4fc5fc6_params(mat_path, prefix=''):
    import scipy.io as sio
    import numpy as np
    conv123 = sio.loadmat(mat_path)
    conv123 = conv123['conv123fc456']
    conv1_weight = conv123[0, 0][0]
    conv1_bias = conv123[0, 0][1]
    conv2_weight = conv123[0, 0][2]
    conv2_bias = conv123[0, 0][3]
    conv3_weight = conv123[0, 0][4]
    conv3_bias = conv123[0, 0][5]
    fc4_weight = conv123[0, 0][6]
    fc4_bias = conv123[0, 0][7]
    fc5_weight = conv123[0, 0][8]
    fc5_bias = conv123[0, 0][9]
    fc6_weight = conv123[0, 0][10]
    fc6_bias = conv123[0, 0][11]

    conv1_weight_ = np.transpose(conv1_weight, [3, 2, 0, 1])
    conv1_bias_ = conv1_bias.reshape((96,))
    conv2_weight_ = np.transpose(conv2_weight, [3, 2, 0, 1])
    conv2_bias_ = conv2_bias.reshape((256,))
    conv3_weight_ = np.transpose(conv3_weight, [3, 2, 0, 1])
    conv3_bias_ = conv3_bias.reshape((512,))
    fc4_weight_ = np.transpose(fc4_weight, [3, 2, 0, 1])
    fc4_bias_ = fc4_bias.reshape((512,))
    fc5_weight_ = np.transpose(fc5_weight, [3, 2, 0, 1])
    fc5_bias_ = fc5_bias.reshape((512,))
    fc6_weight_ = np.transpose(fc6_weight, [3, 2, 0, 1])
    fc6_bias_ = fc6_bias.reshape((2,))

    arg_params = dict()
    arg_params[prefix + 'conv1_weight'] = conv1_weight_
    arg_params[prefix + 'conv1_bias'] = conv1_bias_
    arg_params[prefix + 'conv2_weight'] = conv2_weight_
    arg_params[prefix + 'conv2_bias'] = conv2_bias_
    arg_params[prefix + 'conv3_weight'] = conv3_weight_
    arg_params[prefix + 'conv3_bias'] = conv3_bias_
    arg_params[prefix + 'fc4_weight'] = fc4_weight_
    arg_params[prefix + 'fc4_bias'] = fc4_bias_
    arg_params[prefix + 'fc5_weight'] = fc5_weight_
    arg_params[prefix + 'fc5_bias'] = fc5_bias_
    arg_params[prefix + 'score_weight'] = fc6_weight_
    arg_params[prefix + 'score_bias'] = fc6_bias_
    return arg_params


def init_model(args):
    sym = csym.get_mdnet()
    fixed_param_names = []
    for i in range(1, args.fixed_conv + 1):
        fixed_param_names.append('conv' + str(i) + '_weight')
        fixed_param_names.append('conv' + str(i) + '_bias')

    # fixed_param_names.append('fc4_weight')
    # fixed_param_names.append('fc4_bias')
    model = mx.mod.Module(symbol=sym, context=mx.gpu(0), data_names=('feat_bbox', 'image_patch'),
                          label_names=('label',),
                          fixed_param_names=fixed_param_names)
    t = time()
    model.bind([mx.io.DataDesc('feat_bbox', (1, 128, 5)), mx.io.DataDesc('image_patch', (1, 3, 219, 219))],
               [mx.io.DataDesc('label', (1, 128))])
    print('| mod.bind, cost:%.6f' % (time() - t))

    t = time()
    # conv123fc4fc5 = get_mdnet_conv123fc4fc5fc6_params(
    #     mat_path=args.ROOT_path + '/saved/mdnet_otb-vot15_in_py_for_conv123fc456.mat')
    if args.saved_fname.endswith('shared'):
        model.load_params(os.path.join(args.ROOT_path, args.saved_fname))
    else:
        conv123fc4fc5 = get_mdnet_conv123fc4fc5fc6_params(
            mat_path=args.ROOT_path + '/saved/mdnet_otb-vot15_in_py_for_conv123fc456.mat')
        for k in conv123fc4fc5.keys():
            conv123fc4fc5[k] = mx.ndarray.array(conv123fc4fc5.get(k))
        model.init_params(initializer=mx.initializer.Constant(-0.1), arg_params=conv123fc4fc5, allow_missing=True,
                          force_init=False, allow_extra=True)

    print('| init params, cost:%.6f' % (time() - t))
    return model
