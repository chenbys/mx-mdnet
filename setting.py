from easydict import EasyDict as edict

constant = edict()
constant.stride = 8.
constant.recf = 43.
constant.patch_W = 219.
constant.patch_H = 219.
constant.feat_W = 23.
constant.feat_H = 23.
constant.HWN2NHW = (2, 0, 1)
constant.NHW2HWN = (1, 2, 0)

constant.P_DEBUG = 0
constant.P_TEST = 1
constant.P_RUN = 2
constant.P_FORBID = 3

config = edict()

default = edict()
