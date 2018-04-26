# -*-coding:utf- 8-*-

import numpy as np
import mxnet as mx
import sample
import os
from scipy.misc import imresize
import copy
import util
from extend import get_mdnet_conv123fc4fc5fc6_params
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
    A = list()
    B = list()
    C = list()
    # 伪造一些不准确的pre_region
    pre_regions = []
    for i in np.arange(0.7, 2, 0.1):
        pre_regions.append(util.central_bbox(region, 0, 0, i + 0.3, i - 0.2))
        pre_regions.append(util.central_bbox(region, 0, 0, i - 0.2, i + 0.3))
        pre_regions.append(util.central_bbox(region, 0, 0, i, i))
    for i in np.arange(0.7, 2, 0.1):
        pre_regions.append(util.central_bbox(region, 0, -1, i + 0.1, i - 0.1))
        pre_regions.append(util.central_bbox(region, -1, 0, i - 0.1, i + 0.1))
        pre_regions.append(util.central_bbox(region, 1, 0, i - 0.1, i + 0.1))
        pre_regions.append(util.central_bbox(region, 0, 1, i + 0.1, i - 0.1))
    for i in np.arange(0.7, 2, 0.1):
        pre_regions.append(util.central_bbox(region, 0, -2, i + 0.1, i - 0.1))
        pre_regions.append(util.central_bbox(region, -2, 0, i - 0.1, i + 0.1))
        pre_regions.append(util.central_bbox(region, 2, 0, i - 0.1, i + 0.1))
        pre_regions.append(util.central_bbox(region, 0, 2, i + 0.1, i - 0.1))
    for i in np.arange(0.7, 2, 0.1):
        pre_regions.append(util.central_bbox(region, 0, -0.5, i + 0.3, i - 0.2))
        pre_regions.append(util.central_bbox(region, -0.5, 0, i - 0.2, i + 0.3))
        pre_regions.append(util.central_bbox(region, 0.5, 0, i - 0.2, i + 0.3))
        pre_regions.append(util.central_bbox(region, 0, 0.5, i + 0.3, i - 0.2))

    for pr in pre_regions:
        img_patch, restore_info = util.get_img_patch(img, pr)
        X, Y, W, H, patch_W, patch_H = restore_info
        patch_feat_bbox = util.img2feat(util.xywh2x1y1x2y2(np.array([[0, 0, patch_W, patch_H]])))
        label_patch_bbox = np.array(util.transform_bbox(region, restore_info))
        ideal_patch_bbox = copy.deepcopy(label_patch_bbox)
        ideal_patch_bbox[2], ideal_patch_bbox[3] = max(ideal_patch_bbox[2], 45), max(ideal_patch_bbox[3], 45)

        # if util.bbox_contain([X, Y, W, H], region):
        # 抽样区域包括gt,可以采集正样本
        ideal_feat_bbox = util.img2feat(util.xywh2x1y1x2y2(np.array([ideal_patch_bbox])))[0, :]
        pos_feat_bboxes = sample.get_pos_feat_bboxes(ideal_feat_bbox=ideal_feat_bbox,
                                                     feat_size=patch_feat_bbox[0, 2:] + 1)
        neg_feat_bboxes = sample.get_neg_feat_bboxes(ideal_feat_bbox=ideal_feat_bbox,
                                                     feat_size=patch_feat_bbox[0, 2:] + 1)
        all_feat_bboxes = np.array(pos_feat_bboxes + neg_feat_bboxes)
        # 还原到patch上，以便获得label
        patch_bboxes = util.feat2img(all_feat_bboxes[:, 1:])
        rat = util.overlap_ratio(label_patch_bbox, patch_bboxes)

        pos_samples = all_feat_bboxes[rat > const.train_pos_th, :]
        neg_samples = all_feat_bboxes[rat < const.train_neg_th, :]
        feat_bboxes, labels = np.vstack((pos_samples, neg_samples)), \
                              np.hstack((np.ones((len(pos_samples),)), np.zeros((len(neg_samples)))))

        img_patch = img_patch.transpose(const.HWN2NHW)
        A.append(img_patch)
        B.append(feat_bboxes)
        C.append(labels)

    return A, B, C


def get_update_data(img, gt, regions):
    '''
        原版mdnet每一帧采50 pos 200 neg
        返回该帧构造出的 9 个img_patch, each 16 pos 32 neg
        采集负样本就不太需要缩放。
    :param img_patch:
    :param gt:
    :return:
    '''

    A = list()
    B = list()
    C = list()
    pre_regions = []
    # pre_regions.append(util.central_bbox(gt, 0, 0, 0.6, 0.6))
    pre_regions.append(util.central_bbox(gt, 0, 0, 0.8, 0.8))
    # pre_regions.append(util.central_bbox(gt, 0, 0, 1.2, 1.2))
    pre_regions.append(util.central_bbox(gt, 0, 0, 1.7, 1.7))

    pre_regions.append(util.central_bbox(gt, 0, 0, 0.7, 1))
    pre_regions.append(util.central_bbox(gt, 0, 0, 1, 0.7))
    pre_regions.append(util.central_bbox(gt, 0, 0, 1, 1.5))
    pre_regions.append(util.central_bbox(gt, 0, 0, 1.5, 1))

    pre_regions.append(util.central_bbox(gt, 1, 0, 1, 1))
    pre_regions.append(util.central_bbox(gt, -1, 0, 1, 1))
    pre_regions.append(util.central_bbox(gt, 0, 1, 1, 1))
    pre_regions.append(util.central_bbox(gt, 0, -1, 1, 1))

    pre_regions += regions[-2:]
    const.update_batch_num = 10 + 2

    for pr in pre_regions:
        img_patch, restore_info = util.get_img_patch(img, pr)
        X, Y, W, H, patch_W, patch_H = restore_info
        patch_feat_bbox = util.img2feat(util.xywh2x1y1x2y2(np.array([[0, 0, patch_W, patch_H]])))
        label_patch_bbox = np.array(util.transform_bbox(gt, restore_info))
        ideal_patch_bbox = copy.deepcopy(label_patch_bbox)
        ideal_patch_bbox[2], ideal_patch_bbox[3] = max(ideal_patch_bbox[2], 45), max(ideal_patch_bbox[3], 45)

        # if util.bbox_contain([X, Y, W, H], region):
        # 抽样区域包括gt,可以采集正样本
        ideal_feat_bbox = util.img2feat(util.xywh2x1y1x2y2(np.array([ideal_patch_bbox])))[0, :]
        pos_feat_bboxes = sample.get_pos_feat_bboxes(ideal_feat_bbox=ideal_feat_bbox,
                                                     feat_size=patch_feat_bbox[0, 2:] + 1)
        neg_feat_bboxes = sample.get_neg_feat_bboxes(ideal_feat_bbox=ideal_feat_bbox,
                                                     feat_size=patch_feat_bbox[0, 2:] + 1)
        all_feat_bboxes = np.array(pos_feat_bboxes + neg_feat_bboxes)
        # 还原到patch上，以便获得label
        patch_bboxes = util.feat2img(all_feat_bboxes[:, 1:])
        rat = util.overlap_ratio(label_patch_bbox, patch_bboxes)

        pos_samples = all_feat_bboxes[rat > const.update_pos_th, :]

        neg_samples = all_feat_bboxes[rat < const.update_neg_th, :]
        if pos_samples.shape[0] > 50:
            pos_sel_idx = sample.rand_sample(np.arange(pos_samples.shape[0]), 50)
            pos_samples = pos_samples[pos_sel_idx]
        if neg_samples.shape[0] > 300:
            neg_sel_idx = sample.rand_sample(np.arange(neg_samples.shape[0]), 300)
            neg_samples = neg_samples[neg_sel_idx]

        feat_bboxes, labels = np.vstack((pos_samples, neg_samples)), \
                              np.hstack((np.ones((len(pos_samples),)), np.zeros((len(neg_samples)))))

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

    X, Y, W, H, patch_W, patch_H = restore_info
    patch_feat_bbox = util.img2feat(util.xywh2x1y1x2y2(np.array([[0, 0, patch_W, patch_H]])))
    t = util.transform_bbox(pre_region, restore_info)
    ideal_feat_bbox = util.img2feat(util.xywh2x1y1x2y2(np.array([t])))[0, :]
    feat_bbox = sample.get_predict_feat_bboxes(ideal_feat_bbox=ideal_feat_bbox,
                                               feat_size=patch_feat_bbox[0, 2:] + 1)

    # label的值应该不影响predict的输出
    label = np.ones((feat_bbox.shape[0],))
    img_patch = img_patch.transpose(const.HWN2NHW)

    return ([img_patch], [feat_bbox], [label]), restore_info


def get_pre_train_data(img, region):
    '''
        source mdnet : 30 batch, each 32 pos 96 neg
        now : 30*5 batch, each 1 img_patch 32 pos 96 neg


    :param img_path:
    :param region:
    :param stride_w:
    :param stride_h:
    :return:
    '''
    A = list()
    B = list()
    C = list()
    # 伪造一些不准确的pre_region
    pre_regions = []
    for i in np.arange(0.6, 2, 0.05):
        pre_regions.append(util.central_bbox(region, 0, 0, i + 0.3, i - 0.2))
        pre_regions.append(util.central_bbox(region, 0, 0, i - 0.2, i + 0.3))
        pre_regions.append(util.central_bbox(region, 0, 0, i, i))

    for pr in pre_regions:
        img_patch, restore_info = util.get_img_patch(img, pr)
        X, Y, W, H, patch_W, patch_H = restore_info
        patch_feat_bbox = util.img2feat(util.xywh2x1y1x2y2(np.array([[0, 0, patch_W, patch_H]])))
        label_patch_bbox = np.array(util.transform_bbox(region, restore_info))
        ideal_patch_bbox = copy.deepcopy(label_patch_bbox)
        ideal_patch_bbox[2], ideal_patch_bbox[3] = max(ideal_patch_bbox[2], 45), max(ideal_patch_bbox[3], 45)

        # if util.bbox_contain([X, Y, W, H], region):
        # 抽样区域包括gt,可以采集正样本
        ideal_feat_bbox = util.img2feat(util.xywh2x1y1x2y2(np.array([ideal_patch_bbox])))[0, :]
        pos_feat_bboxes = sample.get_pos_feat_bboxes(ideal_feat_bbox=ideal_feat_bbox,
                                                     feat_size=patch_feat_bbox[0, 2:] + 1)
        neg_feat_bboxes = sample.get_neg_feat_bboxes(ideal_feat_bbox=ideal_feat_bbox,
                                                     feat_size=patch_feat_bbox[0, 2:] + 1)
        all_feat_bboxes = np.array(pos_feat_bboxes + neg_feat_bboxes)
        # 还原到patch上，以便获得label
        patch_bboxes = util.feat2img(all_feat_bboxes[:, 1:])
        rat = util.overlap_ratio(label_patch_bbox, patch_bboxes)

        pos_samples = all_feat_bboxes[rat > const.train_pos_th, :]
        neg_samples = all_feat_bboxes[rat < const.train_neg_th, :]
        feat_bboxes, labels = np.vstack((pos_samples, neg_samples)), \
                              np.hstack((np.ones((len(pos_samples),)), np.zeros((len(neg_samples)))))

        img_patch = img_patch.transpose(const.HWN2NHW)
        A.append(img_patch)
        B.append(feat_bboxes)
        C.append(labels)

    return A, B, C


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


class ParamsHelper(object):
    def __init__(self):
        self.branch_params = {}
        conv123fc4fc5 = get_mdnet_conv123fc4fc5fc6_params(
            mat_path='saved/mdnet_otb-vot15_in_py_for_conv123fc456.mat')
        self.init_branch_param = {}
        self.init_branch_param['score_weight'] = mx.ndarray.array(conv123fc4fc5['score_weight'])
        self.init_branch_param['score_bias'] = mx.ndarray.array(conv123fc4fc5['score_bias'])

    def change_params(self, model, seq_name):
        branch_params = copy.deepcopy(self.branch_params.get(seq_name))
        if branch_params is None:
            branch_params = copy.deepcopy(self.init_branch_param)
            model.set_params(branch_params, None,
                             allow_missing=True, force_init=True, allow_extra=False)
            return model
        model.set_params(branch_params, None,
                         allow_missing=True, force_init=True, allow_extra=False)
        return model

    def update_params(self, model, seq_name):
        arg_params, aux_params = model.get_params()
        branch_params = {}
        branch_params['score_weight'] = copy.deepcopy(arg_params['score_weight'])
        branch_params['score_bias'] = copy.deepcopy(arg_params['score_bias'])
        self.branch_params[seq_name] = branch_params

    def save_params(self, model, k, prefix='k'):
        # 保存在params/prefix_k的文件夹中
        saved_path = 'params/%s_%d/' % (prefix, k)
        if os.path.exists(saved_path) is False:
            os.makedirs(saved_path)

        for key, item in self.branch_params.items():
            mx.ndarray.save(saved_path + key, item)

        model.save_params(saved_path + 'shared')
        print 'save to ' + saved_path

    def load_params(self, model, k, prefix='k'):
        saved_path = 'params/%s_%d/' % (prefix, k)
        if os.path.exists(saved_path) is False:
            print 'not exists, did nothing'
            return

        file_names = os.listdir(saved_path)
        for file_name in file_names:
            param_dict = mx.ndarray.load(saved_path + file_name)
            self.branch_params[file_name] = param_dict

        model.load_params(saved_path + 'shared')
        print 'load from ' + saved_path


class OTB_VOT_Helper(object):
    def __init__(self, path='/Users/chenjunjie/workspace/OTB/'):
        self.home_path = path
        self.seq_names = ['BlurCar3', 'BlurCar4', 'Girl', 'Bird1', 'BlurBody', 'BlurCar2', 'BlurFace', 'BlurOwl', 'Box',
                          'Car1', 'Car4',
                          'CarScale', 'Biker', 'ClifBar', 'Crowds', 'Deer', 'DragonBaby', 'Dudek',
                          'Football', 'Human4', 'Human9', 'Ironman', 'Jump', 'Jumping', 'Liquor', 'Panda',
                          'RedTeam', 'Skating1', 'Skiing', 'Surfer', 'Sylvester', 'Trellis', 'Walking', 'Walking2',
                          'Woman', 'Bird2', 'Board', 'Boy', 'Car2', 'Car24', 'Coke', 'Coupon',
                          'Crossing', 'Dancer', 'Dancer2', 'David2', 'David3', 'Dog', 'Dog1', 'Doll', 'FaceOcc1',
                          'FaceOcc2', 'Fish', 'FleetFace', 'Freeman1', 'Gym', 'Human2',
                          'Human7', 'Human8', 'KiteSurf', 'Lemming', 'Man', 'Mhyang', 'MountainBike', 'Rubik', 'Skater',
                          'Skater2', 'Subway', 'Suv', 'Toy', 'Trans', 'Twinnings', 'Vase']
        # 'Diving',Freeman4
        # self.seq_names = ['Girl', 'Bird1', 'BlurBody', 'Car1', 'Car4']
        # self.seq_names = ['Girl', 'Bird1', 'BlurBody', 'BlurCar2', 'BlurFace', 'BlurOwl', 'Box', 'Car1', 'Car4',
        #                   'CarScale', 'Biker', 'ClifBar', 'Crowds', 'David', 'Deer', 'Diving', 'DragonBaby', 'Dudek',
        #                   'Football', 'Freeman4', 'Human4', 'Human9', 'Ironman', 'Jump', 'Jumping', 'Liquor', 'Panda',
        #                   'RedTeam', 'Skating1', 'Skiing', 'Surfer', 'Sylvester', 'Trellis', 'Walking', 'Walking2',
        #                   'Woman', 'Bird2', 'BlurCar3', 'BlurCar4', 'Board', 'Boy', 'Car2', 'Car24', 'Coke', 'Coupon',
        #                   'Crossing', 'Dancer', 'Dancer2', 'David2', 'David3', 'Dog', 'Dog1', 'Doll', 'FaceOcc1',
        #                   'FaceOcc2', 'Fish', 'FleetFace', 'Football1', 'Freeman1', 'Freeman3', 'Gym', 'Human2',
        #                   'Human7', 'Human8', 'KiteSurf', 'Lemming', 'Man', 'Mhyang', 'MountainBike', 'Rubik', 'Skater',
        #                   'Skater2', 'Subway', 'Suv', 'Toy', 'Trans', 'Twinnings', 'Vase']

        self.data_base = {}

    def add_data(self, seq_name):
        img_paths, gts = self.get_seq(seq_name)
        self.data_base[seq_name] = img_paths, gts
        return img_paths, gts

    def get_data(self, seq_name, frame_idx):
        seq_data = self.data_base.get(seq_name)
        if seq_data == None:
            seq_data = self.add_data(seq_name)
        img_paths, gts = seq_data
        num = len(img_paths)
        return img_paths[frame_idx % num], gts[frame_idx % num]

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

        img_dir_path = os.path.join(self.home_path, seq_name, 'img')
        jpg_files = [jpg_file for jpg_file in os.listdir(img_dir_path)
                     if jpg_file.endswith('.jpg')]
        fill_len = jpg_files[0].__len__() - 4

        img_nums = [int(num.replace('.jpg', '')) for num in jpg_files]
        img_nums = sorted(img_nums)
        img_paths = [os.path.join(img_dir_path, (str(img_num).zfill(fill_len) + '.jpg')) for img_num in img_nums]
        assert len(img_paths) == len(gts), 'num of jpg file and gt bbox must equal: %s' % seq_name
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
