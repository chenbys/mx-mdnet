import os
import cv2
import numpy as np


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
