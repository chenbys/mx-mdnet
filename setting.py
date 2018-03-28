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

const.pred_patch_W = 331.
const.pred_patch_H = 331.
const.pred_feat_W = 37.
const.pred_feat_H = 37.
const.pred_feat_size = [37., 37.]
const.pred_ideal_feat_bbox = [14, 14, 22, 22]

const.HWN2NHW = (2, 0, 1)
const.NHW2HWN = (1, 2, 0)
const.long_term = 100
const.short_term = 20

const.P_DEBUG = 0
const.P_TEST = 1
const.P_RUN = 2
const.P_FORBID = 3

const.update_pos_th = 0.7
const.update_neg_th = 0.3
const.train_pos_th = 0.7
const.train_neg_th = 0.5
