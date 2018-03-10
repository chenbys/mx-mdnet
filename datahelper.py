# -*-coding:utf- 8-*-

import numpy as np
import mxnet as mx
import sample
import os
import cv2
from scipy.misc import imresize
import matplotlib.pyplot as plt

import util


def get_train_data(img_path, region, stride_w=0.1, stride_h=0.1):
    img = cv2.imread(img_path)
    img_H, img_W, c = np.shape(img)
    img_pad = np.concatenate((img, img, img), 0)
    img_pad = np.concatenate((img_pad, img_pad, img_pad), 1)

    x, y, w, h = region
    X, Y, W, H = x - w / 2., y - h / 2., 2 * w, 2 * h
    patches = list()
    for scale_w in np.arange(0.5, 1.6, stride_w):
        for scale_h in np.arange(0.5, 1.6, stride_h):
            W_, H_ = W * scale_w, H * scale_h
            X_, Y_ = x + w / 2. - W_ / 2., y + h / 2. - H_ / 2.
            patches.append([X_, Y_, W_, H_])

    image_patches = list()
    feat_bboxes = list()
    labels = list()
    for patch in patches:
        # crop image as train_data
        # 我的
        X, Y, W, H = patch
        img_patch = imresize(img_pad[int(Y + img_H):int(Y + img_H + H), int(X + img_W):int(X + img_W + W), :],
                             [227, 227])
        # ISSUE: change HWC to CHW
        img_patch = img_patch.reshape((3, 227, 227))

        # get region
        patch_gt = np.array([[227. * (x - X) / W, 227. * (y - Y) / H, 227. * w / W, 227. * h / H]])
        feat_bbox, label = sample.get_01samples(patch_gt)
        image_patches.append(img_patch)
        feat_bboxes.append(feat_bbox)
        labels.append(label)

    return image_patches, feat_bboxes, labels


def get_train_iter(train_data):
    image_patches, feat_bboxes, labels = train_data
    return mx.io.NDArrayIter({'image_patch': image_patches, 'feat_bbox': feat_bboxes}, {'label': labels},
                             batch_size=1, data_name=('image_patch', 'feat_bbox',), label_name=('label',))


def get_predict_data(img_path, pre_region):
    '''

    :param img_path:
    :param pre_region:
    :return: pred_data and restore_info,
    restore_info include the XYWH of img_patch respect to
    '''
    feat_bbox = sample.sample_on_feat(1, 1, 2, 2)
    img = cv2.imread(img_path)
    x, y, w, h = pre_region
    img_H, img_W, c = np.shape(img)
    img_pad = np.concatenate((img, img, img), 0)
    img_pad = np.concatenate((img_pad, img_pad, img_pad), 1)
    W, H = 227 / 131. * w, 227 / 131. * h
    X, Y = img_W + x + w / 2. - W / 2., img_H + y + h / 2. - H / 2.

    img_patch = img_pad[int(Y):int(Y + H), int(X):int(X + W), :]
    img_patch = imresize(img_patch, [227, 227])
    img_patch = img_patch.reshape((3, 227, 227))
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
