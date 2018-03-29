# -*-coding:utf- 8-*-

import numpy as np
import mxnet as mx
import sample
import os
from scipy.misc import imresize

import util
from setting import const
import matplotlib.pyplot as plt


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
    img_H, img_W, c = np.shape(img)
    img_pad = np.concatenate((img, img, img), 0)
    img_pad = np.concatenate((img_pad, img_pad, img_pad), 1)

    x, y, w, h = region
    X, Y, W, H = x - w / 2., y - h / 2., 2 * w, 2 * h
    patches = list()

    for scale_w, scale_h in zip([0.8, 1, 1.4, 1, 2, 2, 1.5, 1.8, 1, 2, 1.2],
                                [0.8, 1, 1.4, 1, 2, 1.5, 2, 1.8, 2, 1.2, 2]):
        W_, H_ = W * scale_w, H * scale_h
        X_, Y_ = x + w / 2. - W_ / 2., y + h / 2. - H_ / 2.
        patches.append([int(X_), int(Y_), int(W_), int(H_)])

    #
    W_, H_ = const.patch_W / 107. * w, const.patch_H / 107. * h
    X_, Y_ = x + w / 2. - W_ / 2., y + h / 2. - H_ / 2.
    patches.append([int(X_), int(Y_), int(W_), int(H_)])
    patches.append([int(X_), int(Y_), int(W_), int(H_)])
    patches.append([int(X_), int(Y_), int(W_), int(H_)])

    image_patches = list()
    feat_bboxes = list()
    labels = list()
    for patch in patches:
        X, Y, W, H = patch
        img_patch = imresize(img_pad[int(Y + img_H):int(Y + img_H + H), int(X + img_W):int(X + img_W + W), :],
                             [int(const.patch_H), int(const.patch_W)])
        img_patch = img_patch.transpose(const.HWN2NHW)

        # get region
        patch_gt = np.array([[const.patch_W * (x - X) / W, const.patch_H * (y - Y) / H,
                              const.patch_W * w / W, const.patch_H * h / H]])
        feat_bbox, label = sample.get_train_samples(patch_gt)
        image_patches.append(img_patch)
        feat_bboxes.append(feat_bbox)
        labels.append(label)

    return image_patches, feat_bboxes, labels


def get_update_data(img, gt):
    '''
        原版mdnet每一帧采50 pos 200 neg
        返回该帧构造出的 9 个img_patch, each 16 pos 32 neg
    :param img_patch:
    :param gt:
    :return:
    '''
    img_H, img_W, c = np.shape(img)
    img_pad = np.concatenate((img, img, img), 0)
    img_pad = np.concatenate((img_pad, img_pad, img_pad), 1)

    x, y, w, h = gt
    X, Y, W, H = x - w / 2., y - h / 2., 2 * w, 2 * h
    patches = list()
    for scale_w, scale_h in zip([0.8, 1, 1.4, 1, 2, 1.7, 1, 2],
                                [0.8, 1, 1.4, 1, 2, 1.7, 2, 1.8]):
        W_, H_ = W * scale_w, H * scale_h
        X_, Y_ = x + w / 2. - W_ / 2., y + h / 2. - H_ / 2.
        patches.append([int(X_), int(Y_), int(W_), int(H_)])

    image_patches = list()
    feat_bboxes = list()
    labels = list()
    for patch in patches:
        # crop image as train_data
        X, Y, W, H = patch
        img_patch = imresize(img_pad[int(Y + img_H):int(Y + img_H + H), int(X + img_W):int(X + img_W + W), :],
                             [int(const.patch_H), int(const.patch_W)])
        # ISSUE: change HWC to CHW
        img_patch = img_patch.transpose(const.HWN2NHW)

        # get region
        patch_gt = np.array([[const.patch_W * (x - X) / W, const.patch_H * (y - Y) / H,
                              const.patch_W * w / W, const.patch_H * h / H]])
        feat_bbox, label = sample.get_update_samples(patch_gt, 30, 120)

        image_patches.append(img_patch)
        feat_bboxes.append(feat_bbox)
        labels.append(label)
    return image_patches, feat_bboxes, labels


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
