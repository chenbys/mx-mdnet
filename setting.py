from easydict import EasyDict as edict

const = edict()
const.stride = 8.
const.recf = 43.
const.patch_W = 219.
const.patch_H = 219.
const.feat_W = 23.
const.feat_H = 23.
const.HWN2NHW = (2, 0, 1)
const.NHW2HWN = (1, 2, 0)
const.long_term = 100
const.short_term = 20

const.P_DEBUG = 0
const.P_TEST = 1
const.P_RUN = 2
const.P_FORBID = 3

config = edict()

default = edict()
