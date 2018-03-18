# -*-coding:utf- 8-*-
import random

import numpy as np
import mxnet as mx
import sample
import os
import cv2
from scipy.misc import imresize
import matplotlib.pyplot as plt

import util


def get_train_data(img_path, gt):
    '''
        one patch one sample one label
    :param img_path:
    :param gt:
    :param stride_w:
    :param stride_h:
    :return:
    '''
    img = cv2.imread(img_path)
    img_H, img_W, c = np.shape(img)
    img_pad = np.concatenate((img, img, img), 0)
    img_pad = np.concatenate((img_pad, img_pad, img_pad), 1)

    def f(img_bbox, img_pad):
        '''
            把img_bbox缩放到107,107,返回203,203的img_patch
        :param img_bbox: x,y,w,h
        :return:
        '''
        x, y, w, h = img_bbox
        W, H = int(203 * w / 107.), int(203 * h / 107.)
        X, Y = int(x + w / 2. - W / 2.), int(y + h / 2. - H / 2.)
        img_patch = img_pad[img_H + Y:img_H + Y + H, img_W + X:img_W + X + W, :]
        img_patch = imresize(img_patch, [203, 203])
        img_patch = img_patch.reshape((3, 203, 203))
        return img_patch

    img_bboxes = []
    xg, yg, wg, hg = gt
    for dx in np.arange(-0.5, 0.5, 0.05):
        for dy in np.arange(-0.5, 0.5, 0.05):
            for dw in np.arange(0.7, 1.4, 0.1):
                for dh in np.arange(0.7, 1.4, 0.1):
                    x = int(max(0, xg + dx * wg))
                    y = int(max(0, yg + dy * hg))
                    w = int(min(wg * dw, img_W - x))
                    h = int(min(hg * dh, img_H - y))
                    img_bboxes.append([x, y, w, h])
    # for dx in np.arange(-1, 1, 0.1):
    #     for dy in np.arange(-1, 1, 0.1):
    #         for dw in np.arange(0.5, 2, 0.1):
    #             for dh in np.arange(0.5, 2, 0.1):
    #                 x = int(max(0, xg + dx * wg))
    #                 y = int(max(0, yg + dy * hg))
    #                 w = int(min(wg * dw, img_W - x))
    #                 h = int(min(hg * dh, img_H - y))
    #                 img_bboxes.append([x, y, w, h])
    rat = util.overlap_ratio(img_bboxes, gt)
    img_bboxes = np.array(img_bboxes)
    pos_img_bboxes = img_bboxes[rat > 0.7, :]
    neg_img_bboxes = img_bboxes[rat < 0.5, :]
    pos_img_bboxes = pos_img_bboxes[sample.rand_sample(np.arange(0, pos_img_bboxes.shape[0]), 500), :]
    neg_img_bboxes = neg_img_bboxes[sample.rand_sample(np.arange(0, neg_img_bboxes.shape[0]), 5000), :]

    pos_idx = np.array(random.sample(range(0, 1000), 960)) % 500
    neg_idx = np.array(random.sample(range(0, 5000), 2880))
    img_patches, feat_bboxes, labels = [], [], []
    for b in range(30):
        for p in range(32):
            idx = pos_idx[b * 32 + p]
            img_patches.append(f(pos_img_bboxes[idx, :], img_pad))
            feat_bboxes.append([p, 6, 6, 16, 16])
            labels.append(1)
        for n in range(96):
            idx = neg_idx[b * 96 + p]
            img_patches.append(f(neg_img_bboxes[idx, :], img_pad))
            feat_bboxes.append([n + 32, 6, 6, 16, 16])
            labels.append(0)

    def check(i):
        img_patch = img_patches[i]
        plt.imshow(img_patch.reshape((203, 203, 3)))
        return labels[i]

    return img_patches, feat_bboxes, labels


def get_train_iter(train_data):
    image_patches, feat_bboxes, labels = train_data
    return mx.io.NDArrayIter({'image_patch': image_patches, 'feat_bbox': feat_bboxes}, {'label': labels},
                             batch_size=128, data_name=('image_patch', 'feat_bbox',), label_name=('label',))


def get_predict_data(img_path, pre_region):
    '''

    :param img_path:
    :param pre_region:
    :return: pred_data and restore_info,
    restore_info include the XYWH of img_patch respect to
    '''
    feat_bbox = sample.get_predict_feat_sample()
    img = cv2.imread(img_path)
    x, y, w, h = pre_region
    img_H, img_W, c = np.shape(img)
    img_pad = np.concatenate((img, img, img), 0)
    img_pad = np.concatenate((img_pad, img_pad, img_pad), 1)
    W, H = 195 / 107. * w, 195 / 107. * h
    X, Y = img_W + x + w / 2. - W / 2., img_H + y + h / 2. - H / 2.

    img_patch = img_pad[int(Y):int(Y + H), int(X):int(X + W), :]
    img_patch = imresize(img_patch, [195, 195])
    img_patch = img_patch.reshape((3, 195, 195))
    # label的值应该不影响predict的输出，设为gt方便调试
    label = np.zeros((feat_bbox.shape[0],))

    return (img_patch, feat_bbox, label), (img_W, img_H, X, Y, W, H)


def get_predict_iter(predict_data):
    img_patch, feat_bbox, label = predict_data
    return mx.io.NDArrayIter({'image_patch': [img_patch], 'feat_bbox': [feat_bbox]}, {'label': [label]},
                             batch_size=1, data_name=('image_patch', 'feat_bbox'), label_name=('label',))


def get_val_data(img_path, pre_region, gt):
    pred_data, restore_info = get_predict_data(img_path, pre_region)
    img_patch, feat_bboxes, label = pred_data
    patch_bboxes = util.feat2img(feat_bboxes[:, 1:])
    img_bboxes = util.restore_img_bbox(patch_bboxes, restore_info)
    rat = util.overlap_ratio(gt, img_bboxes)
    return img_patch, feat_bboxes, rat


def get_val_iter(val_data):
    img_patch, feat_bbox, label = val_data
    return mx.io.NDArrayIter({'image_patch': [img_patch], 'feat_bbox': [feat_bbox]}, {'label': [label]},
                             batch_size=1, data_name=('image_patch', 'feat_bbox'), label_name=('label',))


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
            size = np.shape(cv2.imread(self.path + seq_name + '/img/' + img_paths[0]))
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
        return rep_times, get_train_iter(get_train_data(img_path, gt)), get_train_iter(get_train_data(val_path, val_gt))
