from easydict import EasyDict as edict

const = edict()
const.stride = 8.
const.recf = 43.
const.patch_W = 219.
const.patch_H = 219.
const.ideal_feat_bbox = [7, 7, 15, 15]
const.feat_W = 23.
const.feat_H = 23.
const.feat_size = [23., 23.]

const.HWN2NHW = (2, 0, 1)
const.NHW2HWN = (1, 2, 0)
const.long_term = 100
const.short_term = 20

const.P_DEBUG = 0
const.P_TEST = 1
const.P_RUN = 2
const.P_FORBID = 3

config = edict()
config.update_pos_th = 0.7
config.update_neg_th = 0.3
config.train_pos_th = 0.7
config.train_neg_th = 0.5
