from dataset.OTBHelper import OTBHelper
import numpy as np

oh = OTBHelper()
gts = oh.get_gts()
infos = oh.get_seq_infos()

seq_names = oh.seq_names
# analyse w,h
wh_max = list()
wh_min = list()
wh_mean = list()
img_h = list()
img_w = list()
dxs=list()
dys=list()
dws=list()
dhs=list()

for seq_name in seq_names:
    gt = gts.get(seq_name)

    length, (h, w, c) = infos.get(seq_name)
    img_h.append(h)
    img_w.append(w)
    wh_max.append(np.max(gt, 0))
    wh_min.append(np.min(gt, 0))
    wh_mean.append(np.mean(gt, 0))

# analyse they with img_w,img_h

# analyse
