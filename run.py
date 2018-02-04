import mxnet as mx
import numpy as np
import sample
import datahelper
import util
import extend
from setting import config


def track(args, model, img_path, pre_region):
    feat_bboxes = sample.sample_on_feat()
    img_patch, predict_iter = datahelper.get_predict_iter(img_path, pre_region, feat_bboxes, True)

    def check_pred(i):
        # R = res.asnumpy()
        # R[i]
        feat_bbox = feat_bboxes[i, 1:].reshape(1, 4)
        img_bbox = util.feat2img(feat_bbox).reshape(4, )
        img_patch_ = np.reshape(img_patch, (227, 227, 3))
        import matplotlib.pyplot as plt
        from matplotlib import patches

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img_patch_)
        ax.add_patch(patches.Rectangle((img_bbox[0], img_bbox[1]), img_bbox[2], img_bbox[3],
                                       linewidth=2, edgecolor='y', facecolor='none'))
        fig.show()
        return fig

    # check_pred(0)
    res = model.predict(predict_iter)

    if args.loss_type == 0:
        res = res[:, 1]
    elif args.loss_type == 1:
        res = res
    opt_idx = mx.ndarray.topk(res, k=5).asnumpy().astype('int32')
    opt_feat_bboxes = feat_bboxes[opt_idx, 1:]
    opt_img_bboxes = util.feat2img(opt_feat_bboxes)
    opt_img_bbox = opt_img_bboxes.mean(0)
    return opt_img_bbox


def run_test(args, model=None):
    vot = datahelper.VOTHelper(args.VOT_path)
    img_list, gts = vot.get_seq('bag')
    print img_list[0]
    print gts[0]
    v0 = datahelper.get_train_iter(datahelper.get_train_data())
    v1 = datahelper.get_train_iter(datahelper.get_train_data(img_list[1], gts[1]))
    v2 = datahelper.get_train_iter(datahelper.get_train_data(img_list[2], gts[2]))
    if model is None:
        model = extend.init_model(loss_type=1, fixed_conv=0, load_conv123=False, saved_fname='saved/test')
    r0 = model.score(v0, extend.MDNetIOUACC())
    r1 = model.score(v1, extend.MDNetIOUACC())
    r2 = model.score(v2, extend.MDNetIOUACC())
    print r0
    print r1
    print r2


if __name__ == '__main__':
    config.ctx = mx.cpu(0)
    # config.ctx = mx.gpu(2)
    # vot = datahelper.VOTHelper('/home/chenjunjie/dataset/VOT2015')

    import easydict

    # args = easydict.EasyDict()
    # args.loss_type = 1
    # img_path = img_list[1]
    # pre_region = gts[0]
    # track(args, model, img_path, pre_region)
