# -*-coding:utf- 8-*-

import numpy as np
import mxnet as mx
import sample
import os
from scipy.misc import imresize

import util
from setting import const
import matplotlib.pyplot as plt
import kit


def get_train_data(img, region):
    '''
        source mdnet : 30 batch, each 32 pos 96 neg
        now : 30*5 batch, each 1 img_patch 32 pos 96 neg


    :param img_path:
    :param region:
    :param stride_w:
    :param stride_h:
    :return:
    '''
    pos_sample_num, neg_sample_num = 32, 96
    img_H, img_W, c = np.shape(img)

    A = list()
    B = list()
    C = list()
    # 伪造一些不准确的pre_region
    pre_regions = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for ws in [0.5, 0.7, 1, 1.5, 2]:
                for hs in [0.5, 0.7, 1, 1.5, 2]:
                    pre_regions.append(util.central_bbox(region, dx, dy, ws, hs, img_W, img_H))

    for pr in pre_regions:
        img_patch, restore_info = util.get_img_patch(img, pr)
        X, Y, W, H, patch_W, patch_H = restore_info
        patch_feat_bbox = util.img2feat(util.xywh2x1y1x2y2(np.array([[0, 0, patch_W, patch_H]])))
        label_patch_bbox = util.transform_bbox(region, restore_info)

        if util.bbox_contain([X, Y, W, H], region):
            # 抽样区域包括gt,可以采集正样本
            ideal_feat_bbox = util.img2feat(util.xywh2x1y1x2y2(np.array([label_patch_bbox])))[0, :]
            all_feat_bboxes = sample.get_train_feat_bboxes(ideal_feat_bbox=ideal_feat_bbox,
                                                           feat_size=patch_feat_bbox[0, 2:] + 1)
        else:
            # 抽样区域不完全包括gt，很可能无法采集到正样本
            all_feat_bboxes = sample.get_neg_feat_bboxes(feat_size=patch_feat_bbox[0, 2:] + 1)

        # 还原到patch上，以便获得label
        patch_bboxes = util.feat2img(all_feat_bboxes[:, 1:])
        rat = util.overlap_ratio(label_patch_bbox, patch_bboxes)

        # pos
        pos_samples = all_feat_bboxes[rat > const.train_pos_th, :]
        if len(pos_samples) > pos_sample_num / 3.:
            pos_select_index = sample.rand_sample(np.arange(0, pos_samples.shape[0]), pos_sample_num)
            neg_samples = all_feat_bboxes[rat < const.train_neg_th, :]
            neg_select_index = sample.rand_sample(np.arange(0, neg_samples.shape[0]), neg_sample_num)
            feat_bboxes, labels = np.vstack((pos_samples[pos_select_index], neg_samples[neg_select_index])), \
                                  np.hstack((np.ones((pos_sample_num,)), np.zeros((neg_sample_num,))))
        else:
            # 没采集到足够的正样本
            neg_samples = all_feat_bboxes[rat < const.train_neg_th, :]
            neg_select_index = sample.rand_sample(np.arange(0, neg_samples.shape[0]), pos_sample_num + neg_sample_num)
            feat_bboxes, labels = neg_samples[neg_select_index], np.zeros((pos_sample_num + neg_sample_num,))

        img_patch = img_patch.transpose(const.HWN2NHW)
        A.append(img_patch)
        B.append(feat_bboxes)
        C.append(labels)

    return A, B, C


def get_update_data(img, gt):
    '''
        原版mdnet每一帧采50 pos 200 neg
        返回该帧构造出的 9 个img_patch, each 16 pos 32 neg
    :param img_patch:
    :param gt:
    :return:
    '''

    pos_sample_num, neg_sample_num = 16, 64
    img_H, img_W, c = np.shape(img)

    A = list()
    B = list()
    C = list()
    # 伪造一些不准确的pre_region
    pre_regions = []
    for dx, dy in zip([-0.5, 0, 0.5], [-0.5, 0, 0.5]):
        for ws, hs in zip([0.5, 0.7, 1, 1.5, 2], [0.5, 0.7, 1, 1.5, 2]):
            pre_regions.append(util.central_bbox(gt, dx, dy, ws, hs, img_W, img_H))

    for pr in pre_regions:
        img_patch, restore_info = util.get_img_patch(img, pr)
        X, Y, W, H, patch_W, patch_H = restore_info
        patch_feat_bbox = util.img2feat(util.xywh2x1y1x2y2(np.array([[0, 0, patch_W, patch_H]])))
        label_patch_bbox = util.transform_bbox(gt, restore_info)

        if util.bbox_contain([X, Y, W, H], gt):
            # 抽样区域包括gt,可以采集正样本
            ideal_feat_bbox = util.img2feat(util.xywh2x1y1x2y2(np.array([label_patch_bbox])))[0, :]
            all_feat_bboxes = sample.get_train_feat_bboxes(ideal_feat_bbox=ideal_feat_bbox,
                                                           feat_size=patch_feat_bbox[0, 2:] + 1)
        else:
            # 抽样区域不完全包括gt，很可能无法采集到正样本
            all_feat_bboxes = sample.get_neg_feat_bboxes(feat_size=patch_feat_bbox[0, 2:] + 1)

        # 还原到patch上，以便获得label
        patch_bboxes = util.feat2img(all_feat_bboxes[:, 1:])
        rat = util.overlap_ratio(label_patch_bbox, patch_bboxes)

        # pos
        pos_samples = all_feat_bboxes[rat > const.train_pos_th, :]
        if len(pos_samples) > pos_sample_num / 3.:
            pos_select_index = sample.rand_sample(np.arange(0, pos_samples.shape[0]), pos_sample_num)
            neg_samples = all_feat_bboxes[rat < const.train_neg_th, :]
            neg_select_index = sample.rand_sample(np.arange(0, neg_samples.shape[0]), neg_sample_num)
            feat_bboxes, labels = np.vstack((pos_samples[pos_select_index], neg_samples[neg_select_index])), \
                                  np.hstack((np.ones((pos_sample_num,)), np.zeros((neg_sample_num,))))
        else:
            # 没采集到足够的正样本
            neg_samples = all_feat_bboxes[rat < const.train_neg_th, :]
            neg_select_index = sample.rand_sample(np.arange(0, neg_samples.shape[0]), pos_sample_num + neg_sample_num)
            feat_bboxes, labels = neg_samples[neg_select_index], np.zeros((pos_sample_num + neg_sample_num,))

        img_patch = img_patch.transpose(const.HWN2NHW)
        A.append(img_patch)
        B.append(feat_bboxes)
        C.append(labels)

    return A, B, C


def get_predict_data(img, pre_region):
    '''

    :param img_path:
    :param pre_region:
    :return: pred_data and restore_info,
    restore_info include the XYWH of img_patch respect to
    '''
    img_patch, restore_info = util.get_img_patch(img, pre_region)
    img_patch = img_patch.transpose(const.HWN2NHW)

    X, Y, W, H, patch_W, patch_H = restore_info
    patch_feat_bbox = util.img2feat(util.xywh2x1y1x2y2(np.array([[0, 0, patch_W, patch_H]])))
    t = util.transform_bbox(pre_region, restore_info)
    ideal_feat_bbox = util.img2feat(util.xywh2x1y1x2y2(np.array([t])))[0, :]
    feat_bbox = sample.get_predict_feat_bboxes(ideal_feat_bbox=ideal_feat_bbox,
                                               feat_size=patch_feat_bbox[0, 2:] + 1)

    # label的值应该不影响predict的输出
    label = np.ones((feat_bbox.shape[0],))

    return ([img_patch], [feat_bbox], [label]), restore_info


def get_iter(data):
    image_patches, feat_bboxes, labels = data
    return mx.io.NDArrayIter({'image_patch': image_patches, 'feat_bbox': feat_bboxes}, {'label': labels},
                             batch_size=1, data_name=('image_patch', 'feat_bbox',), label_name=('label',))


def get_data_batches(data):
    image_patches, feat_bboxes, labels = data
    length = len(labels)
    data_batches = []
    for i in range(length):
        x, y, z = mx.nd.array([image_patches[i]]), mx.nd.array([feat_bboxes[i]]), mx.nd.array([labels[i]])
        data_batches.append(mx.io.DataBatch([y, x], [z]))

    return data_batches


class OTBHelper(object):
    def __init__(self, path='/Users/chenjunjie/workspace/OTB/'):
        self.home_path = path
        self.seq_names = ['Basketball',
                          'Biker',
                          'Bird1',
                          'Bird2',
                          'BlurBody',
                          'BlurCar1',
                          'BlurCar2',
                          'BlurCar3',
                          'BlurCar4',
                          'BlurFace',
                          'BlurOwl',
                          'Board',
                          'Bolt',
                          'Bolt2',
                          'Box',
                          'Boy',
                          'Car1',
                          'Car2',
                          'Car24',
                          'Car4',
                          'CarDark',
                          'CarScale',
                          'ClifBar',
                          'Coke',
                          'Couple',
                          'Coupon',
                          'Crossing',
                          'Crowds',
                          'Dancer',
                          'Dancer2',
                          'David2',
                          'David3',
                          'Deer',
                          'Diving',
                          'Dog',
                          'Dog1',
                          'Doll',
                          'DragonBaby',
                          'Dudek',
                          'FaceOcc1',
                          'FaceOcc2',
                          'Fish',
                          'FleetFace',
                          'Football',
                          'Freeman1',
                          'Girl',
                          'Girl2',
                          'Gym',
                          'Human2',
                          'Human3',
                          'Human4',
                          'Human5',
                          'Human6',
                          'Human7',
                          'Human8',
                          'Human9',
                          'Ironman',
                          'Jump',
                          'Jumping',
                          'KiteSurf',
                          'Lemming',
                          'Liquor',
                          'Man',
                          'Matrix',
                          'Mhyang',
                          'MotorRolling',
                          'MountainBike',
                          'Panda',
                          'RedTeam',
                          'Rubik',
                          'Shaking',
                          'Singer1',
                          'Singer2',
                          'Skater',
                          'Skater2',
                          'Skating1',
                          'Skiing',
                          'Soccer',
                          'Subway',
                          'Surfer',
                          'Suv',
                          'Sylvester',
                          'Tiger1',
                          'Tiger2',
                          'Toy',
                          'Trans',
                          'Trellis',
                          'Twinnings',
                          'Vase',
                          'Walking',
                          'Walking2',
                          'Woman']
        # rm those seq
        # 'David',
        # 'Football1',
        # 'Freeman3',
        # 'Freeman4',
        # 'Jogging.1', 'Skating2.1',
        # 'Jogging.2', 'Skating2.2']

    def get_seq(self, seq_name):
        gt_path = os.path.join(self.home_path, seq_name, 'groundtruth_rect.txt')
        gts = list()
        for line in open(gt_path):
            line = line.replace('\t', ',')
            line = line.replace('\n', '')
            line = line.replace('\r', '')
            line = line.replace(' ', ',')
            x, y, w, h = line.split(',')
            gts.append([float(x), float(y), float(w), float(h)])
        length = len(gts)

        img_paths = [os.path.join(self.home_path, seq_name, 'img', '%0*d.jpg' % (4, idx))
                     for idx in range(1, length + 1)]
        return img_paths, gts

    def get_img(self, seq_name):
        img_root = os.path.join(self.path, seq_name, 'img')
        return [os.path.join(img_root, img_name) for img_name in os.listdir(img_root)]

    def get_gt(self, seq_name, sp=''):
        txt_path = os.path.join(self.path, seq_name, 'groundtruth_rect' + sp + '.txt')
        gt = list()
        for line in open(txt_path):
            line = line.replace('\t', ',')
            line = line.replace('\n', '')
            line = line.replace('\r', '')
            line = line.replace(' ', ',')
            x, y, w, h = line.split(',')
            gt.append([float(x), float(y), float(w), float(h)])

        return gt

    def get_gts(self):
        gts = {}

        for do_name in self.double_names:
            gts[do_name + '.1'] = self.get_gt(do_name, '.1')
            gts[do_name + '.2'] = self.get_gt(do_name, '.2')

        for seq_name in self.seq_names:
            gts[seq_name] = self.get_gt(seq_name)

        return gts

    def get_seq_infos(self):
        info = {}
        for seq_name in self.seq_names:
            path = self.path + seq_name + '/img'
            img_paths = os.listdir(path)
            length = len(img_paths)
            size = np.shape(plt.imread(self.path + seq_name + '/img/' + img_paths[0]))
            info[seq_name] = list([length, size])
        return info


class VOTHelper(object):
    def __init__(self, vot_path='/Users/chenjunjie/workspace/VOT2015'):
        self.home_path = vot_path
        self.seq_names = []
        list_path = os.path.join(self.home_path, 'list.txt')
        for line in open(list_path):
            self.seq_names.append(line.replace('\n', ''))

        self.__load_seq__()

    def __load_seq__(self):
        self.seq_dict = {}
        self.gt_dict = {}
        self.length_dict = {}
        for seq_name in self.seq_names:
            seq, gt = self.get_seq(seq_name)
            self.seq_dict[seq_name] = seq
            self.gt_dict[seq_name] = gt
            self.length_dict[seq_name] = len(gt)

    def get_seq(self, seq_name):
        gt_path = os.path.join(self.home_path, seq_name, 'groundtruth.txt')
        gts = []
        img_paths = []
        frame_idx = 1
        for line in open(gt_path):
            r = line.replace('\n', '').split(',')
            r = [float(x) for x in r]
            x = r[::2]
            y = r[1::2]
            x1, y1, x2, y2 = min(x), min(y), max(x), max(y)
            gts.append([x1, y1, x2 - x1 + 1, y2 - y1 + 1])

            img_name = '%0*d.jpg' % (8, frame_idx)
            img_paths.append(os.path.join(self.home_path, seq_name, img_name))
            frame_idx += 1
        return img_paths, gts

    def get_data(self, k):
        '''
            call before load_seq
        :param k:  in the k-th iter
        :return:    rep_times,train_iter,val_iter
        '''
        seq_idx = k % 60
        seq_name = self.seq_names[seq_idx]
        img_paths, gts, length = self.seq_dict[seq_name], self.gt_dict[seq_name], self.length_dict[seq_name]

        frame_num = k / 60
        rep_times, frame_idx = frame_num / 60, frame_num % length

        img_path, gt = img_paths[frame_idx], gts[frame_idx]
        val_path, val_gt = img_paths[(frame_idx + 1) % length], gts[(frame_idx + 1) % length]
        return rep_times, get_iter(get_train_data(img_path, gt)), get_iter(get_train_data(val_path, val_gt))
