import mxnet as mx
import cv2
import numpy as np
import sample
import datahelper


def track(model, img_path, pre_region):
    img = cv2.imread(img_path)
    feat_bbox = sample.sample_on_feat()
    predict_iter = datahelper.get_predict_iter(img, pre_region, feat_bbox)
    res = model.predict(predict_iter)
    pos_score = res[:, 1]
    opt_idx = mx.ndarray.topk(pos_score, k=5).asnumpy().astype('int32')
    opt_feat_bboxes = feat_bbox[opt_idx, 1:]
    opt_img_bboxes = sample.feat2img(opt_feat_bboxes)
    opt_img_bbox = opt_img_bboxes.mean(0)
    return opt_img_bbox

