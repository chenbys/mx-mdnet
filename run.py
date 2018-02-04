import mxnet as mx
import numpy as np
import sample
import datahelper
import util
import extend
from setting import config
import matplotlib.pyplot as plt
from matplotlib import patches


def track_seq(model, img_paths, gts, topk=5):
    res = []
    length = len(img_paths)
    pre_region = gts[0]
    for T in range(1, length):
        print T
        img_path = img_paths[T]
        pre_region = track(model, img_path, pre_region, topk=topk)
        res.append(pre_region)

    r = np.array(res)
    g = np.array(gts[1:])
    iou = util.overlap_ratio(r, g)
    return res


def track(model, img_path, pre_region, topk=5):
    # only for iou loss
    feat_bboxes = sample.sample_on_feat()
    pred_data, restore_info = datahelper.get_predict_data(img_path, pre_region, feat_bboxes)
    pred_iter = datahelper.get_predict_iter(pred_data)

    def restore_img_bbox(opt_img_bbox, restore_info):
        xo, yo, wo, ho = opt_img_bbox
        img_W, img_H, X, Y, W, H = restore_info
        x, y = W / 227. * xo + X - img_W, H / 227. * yo + Y - img_H
        w, h = W / 227. * wo, H / 227. * ho
        return x, y, w, h

    def check_pred_data(i):
        feat_bbox = feat_bboxes[i, 1:].reshape(1, 4)
        img_bbox = util.feat2img(feat_bbox).reshape(4, )
        img_patch_ = np.reshape(pred_data[0], (227, 227, 3))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img_patch_)
        ax.add_patch(patches.Rectangle((img_bbox[0], img_bbox[1]), img_bbox[2], img_bbox[3],
                                       linewidth=2, edgecolor='y', facecolor='none'))
        fig.show()
        return fig

    # check_pred(0)
    res = model.predict(pred_iter)
    opt_idx = mx.ndarray.topk(res, k=topk).asnumpy().astype('int32')
    opt_feat_bboxes = feat_bboxes[opt_idx, 1:]
    opt_patch_bboxes = util.feat2img(opt_feat_bboxes)
    opt_patch_bbox = opt_patch_bboxes.mean(0)
    opt_img_bbox = restore_img_bbox(opt_patch_bbox, restore_info)

    return opt_img_bbox


if __name__ == '__main__':
    config.ctx = mx.cpu(0)
    # config.ctx = mx.gpu(2)
    # vot = datahelper.VOTHelper('/home/chenjunjie/dataset/VOT2015')
    vot = datahelper.VOTHelper()
    img_list, gts = vot.get_seq('bag')
    # v0 = datahelper.get_train_iter(datahelper.get_train_data(img_list[0], gts[0]))
    # v1 = datahelper.get_train_iter(datahelper.get_train_data(img_list[1], gts[1]))
    # v2 = datahelper.get_train_iter(datahelper.get_train_data(img_list[2], gts[2]))
    model = extend.init_model(loss_type=1, fixed_conv=0, load_conv123=True, saved_fname='saved/finished_3frame')
    # r0 = model.score(v0, extend.MDNetIOUACC())
    # r1 = model.score(v1, extend.MDNetIOUACC())
    # r2 = model.score(v2, extend.MDNetIOUACC())
    res = track_seq(model, img_list[:10], gts[:10])

    T = 3
    img_path = img_list[T - 1]
    pre_region = gts[T - 2]


    def check_pred_res(res, gt):
        img = plt.imread(img_path)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img)
        ax.add_patch(patches.Rectangle((gt[0], gt[1]), gt[2], gt[3],
                                       linewidth=2, edgecolor='blue', facecolor='none'))
        ax.add_patch(patches.Rectangle((res[0], res[1]), res[2], res[3],
                                       linewidth=2, edgecolor='y', facecolor='none'))
        fig.show()


    res = track(model, img_path, pre_region)
    check_pred_res(res, gts[T - 1])
    a = 1
