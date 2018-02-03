import mxnet as mx
import numpy as np
import sample
import datahelper
import util
import extend
from setting import config


def track(args, model, img_path, pre_region):
    feat_bbox = sample.sample_on_feat()
    predict_iter = datahelper.get_predict_iter(img_path, pre_region, feat_bbox)
    res = model.predict(predict_iter)

    if args.loss_type == 0:
        res = res[:, 1]
    elif args.loss_type == 1:
        res = res * -1
    opt_idx = mx.ndarray.topk(res, k=5).asnumpy().astype('int32')
    opt_feat_bboxes = feat_bbox[opt_idx, 1:]
    opt_img_bboxes = util.feat2img(opt_feat_bboxes)
    opt_img_bbox = opt_img_bboxes.mean(0)
    return opt_img_bbox


if __name__ == '__main__':
    config.ctx = mx.cpu(0)

    vot = datahelper.VOTHelper()
    img_list, gts = vot.get_seq('bag')
    val_iter = datahelper.get_train_iter(datahelper.get_train_data(img_list[1], gts[1]))
    img_path = img_list[1]
    pre_region = gts[0]

    model = extend.init_model(0, 0, False, 'saved/finished6frame')
    res = model.score(val_iter, extend.MDNetACC())
    track(model, img_path, pre_region)
