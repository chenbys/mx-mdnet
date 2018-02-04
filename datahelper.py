import numpy as np
import mxnet as mx
import sample
import os
import cv2
from scipy.misc import imresize
import util


def get_train_data(img_path, region, stride_w=0.2, stride_h=0.2, iou_label=True):
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
        X, Y, W, H = patch
        img_patch = imresize(img_pad[int(Y + img_H):int(Y + img_H + H), int(X + img_W):int(X + img_W + W), :],
                             [227, 227])
        # ISSUE: change HWC to CHW
        img_patch = img_patch.reshape((3, 227, 227))

        # get region
        label_region = np.array([[227. * (x - X) / W, 227. * (y - Y) / H, 227. * w / W, 227. * h / H]])
        label_feat = util.x1y2x2y22xywh(util.img2feat(util.xywh2x1y1x2y2(label_region)))
        if iou_label:
            feat_bbox, label = sample.get_samples_with_iou_label(label_feat)
        else:
            feat_bbox, label = sample.get_samples(label_feat)
        image_patches.append(img_patch)
        feat_bboxes.append(feat_bbox)
        labels.append(label)

    return image_patches, feat_bboxes, labels


def get_train_iter(train_data):
    image_patches, feat_bboxes, labels = train_data
    return mx.io.NDArrayIter({'image_patch': image_patches, 'feat_bbox': feat_bboxes}, {'label': labels},
                             batch_size=1, data_name=('image_patch', 'feat_bbox',), label_name=('label',))


def get_predict_data(img_path, pre_region, feat_bbox):
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


class OTBHelper(object):
    def __init__(self, path='/Users/chenjunjie/workspace/OTB/'):
        self.path = path
        self.double_names = ['Jogging', 'Skating2']
        seq_names = ['Basketball',
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
                     'David',
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
                     'Football1',
                     'Freeman1',
                     'Freeman3',
                     'Freeman4',
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
                     'Jogging',
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
                     'Skating2',
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

        for do_name in self.double_names:
            seq_names.remove(do_name)
        self.seq_names = seq_names

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
