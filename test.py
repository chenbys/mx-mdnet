# -*-coding:utf- 8-*-

import numpy as np
from matplotlib import patches

import util
import matplotlib.pyplot as plt
import datahelper
from setting import const



def check_val_data():
    path = '/media/chen/datasets/OTB/Liquor/img/0001.jpg'
    region = [256, 152, 73, 210]
    datahelper.get_update_data(path, region)


def check_fit(patch_idx, sample_idx, img_patches, feat_bboxes, labels, scores):
    img_patches, feat_bboxes, labels, scores = img_patches.asnumpy(), feat_bboxes.asnumpy(), labels.asnumpy(), scores.asnumpy()
    feat_bbox = feat_bboxes[patch_idx, sample_idx, :]
    label = labels[patch_idx, sample_idx]
    score = scores[patch_idx, sample_idx]
    img_patch = img_patches[patch_idx, :, :, :]

    patch_bbox = util.feat2img(feat_bbox[1:].reshape((1, 4))).reshape(4, )
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img_patch.reshape((227, 227, 3)))
    ax.add_patch(patches.Rectangle((patch_bbox[0], patch_bbox[1]), patch_bbox[2], patch_bbox[3],
                                   linewidth=2, edgecolor='red', facecolor='none'))
    fig.show()

    return label, score


def check_fit_plot(labels, scores, patch_idx):
    plt.plot(labels[patch_idx, :])
    plt.plot(scores[patch_idx, :])


def test_sample():
    import cv2
    import matplotlib.pyplot as plt
    from scipy.misc import imresize

    path = '/Users/chenjunjie/workspace/OTB/Liquor/img/0001.jpg'
    img = cv2.imread(path)
    obj = img[152:152 + 210, 256:256 + 73, :]
    plt.imshow(imresize(obj, [211, 211]))
    plt.waitforbuttonpress()


def check_train_data():
    import datahelper
    import util
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    img_path = '/Users/chenjunjie/workspace/OTB/Liquor/img/0001.jpg'
    region = [256, 152, 73, 210]
    img_patches, feat_bboxes_list, labels_list = datahelper.get_train_data(img_path, region)

    def check(patch_i, sample_j):
        img_patch = img_patches[patch_i]
        img_patch_ = img_patch.reshape((227, 227, 3))
        feat_bbox = np.array([feat_bboxes_list[patch_i][sample_j][1:]])
        img_bbox = util.feat2img(feat_bbox)[0, :]
        label = labels_list[patch_i][sample_j]

        print label
        print img_bbox
        print feat_bbox
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img_patch_)
        ax.add_patch(patches.Rectangle((img_bbox[0], img_bbox[1]), img_bbox[2], img_bbox[3],
                                       linewidth=2, edgecolor='y', facecolor='none'))
        return label

    check(16, 350)
    return


def check_train_data_IOU():
    import datahelper
    import util
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    img_path = '/Users/chenjunjie/workspace/OTB/Liquor/img/0001.jpg'
    region = [256, 152, 73, 210]
    img_patches, feat_bboxes_list, labels_list = datahelper.get_train_data(img_path, region, iou_label=True)

    def check(patch_i, sample_j):
        img_patch = img_patches[patch_i]
        img_patch_ = img_patch.reshape((227, 227, 3))
        feat_bbox = np.array([feat_bboxes_list[patch_i][sample_j][1:]])
        img_bbox = util.feat2img(feat_bbox)[0, :]
        label = labels_list[patch_i][sample_j]

        print label
        print img_bbox
        print feat_bbox
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img_patch_)
        ax.add_patch(patches.Rectangle((img_bbox[0], img_bbox[1]), img_bbox[2], img_bbox[3],
                                       linewidth=2, edgecolor='y', facecolor='none'))
        return label

    check(16, 350)
    return


def test_load_net():
    import extend
    import csym
    import mxnet as mx
    import numpy as np

    conv123_param = extend.get_mdnet_conv123_params()
    conv123_param['image_patch'] = np.ones((1, 3, 219, 227)).astype('float')
    loss, conv1, lrn2 = csym.get_mdnet()
    sym = lrn2
    param = dict()
    for k in sym.list_inputs():
        param[k] = mx.ndarray.array(conv123_param.get(k))
    e = sym.bind(mx.cpu(0), param)
    res = e.forward()[0].asnumpy()
    return


def test_train_iter():
    import datahelper
    img_path = '/Users/chenjunjie/workspace/OTB/Liquor/img/0001.jpg'
    region = [256, 152, 73, 210]
    train_data = datahelper.get_train_data(img_path, region, iou_label=True)
    trian_iter = datahelper.get_iter(train_data)
    return


def test_imresize():
    from scipy.misc import imresize
    import kit
    img_path = '/media/chen/datasets/OTB/Liquor/img/0001.jpg'
    img = plt.imread(img_path)
    img = img[150:, 250:, :]
    img_H, img_W, c = np.shape(img)
    region = [256 - 250, 152 - 150, 73, 210]

    def get_patch(img, region):
        x, y, w, h = region
        W, H = const.pred_patch_W / 107. * w, const.pred_patch_H / 107. * h
        W, H = min(img_W, W), min(img_H, H)
        patch_W, patch_H = W * 107. / w, H * 107. / h
        X, Y = max(0, min(img_W - W, x + w / 2. - W / 2.)), max(0, min(img_H - H, y + h / 2. - H / 2.))
        img_patch = img[int(Y):int(Y + H), int(X):int(X + W), :]
        img_patch_ = imresize(img_patch, [int(patch_H), int(patch_W)])
        return img_patch_, [X, Y, W, H, patch_W, patch_H]

    def restore_img(patch_bboxes, resotre_info):
        '''
            X, Y, W, H, patch_W, patch_H = resotre_info

            X,Y,W,H是原图上的bbox，用来resize到patch_W,patch_H大小的

        :param patch_img_bbox:
        :param resotre_info:
        :return:
        '''
        x_ = patch_bboxes[:, 0]
        y_ = patch_bboxes[:, 1]
        w_ = patch_bboxes[:, 2]
        h_ = patch_bboxes[:, 3]

        X, Y, W, H, patch_W, patch_H = resotre_info
        w, h = w_ / patch_W * W, h_ / patch_H * H
        x, y = x_ / patch_W * W, y_ / patch_H * H
        x, y = x + X, y + Y
        img_bboxes = np.vstack((x, y, w, h)).transpose()

        return img_bboxes

    img_path, restore_info = get_patch(img, region)
    b = restore_img(np.array([[10, 10, 107, 107], [10, 10, 100, 100]]), restore_info)
    return


if __name__ == '__main__':
    datahelper.get_predict_data(plt.imread('/media/chen/datasets/OTB/Liquor/img/0001.jpg'), [256, 152, 73, 210])

'''
    # N time for 1 batch
    t1 = time.time()
    for i in range(200):
        res = model.predict(pred_iter).asnumpy()
    t2 = time.time()

    # 1 time for N batch
    [img_patch], [feat_bboxes], [l] = pred_data
    a, b, c = [], [], []
    for i in range(200):
        a.append(img_patch)
        b.append(feat_bboxes)
        c.append(l)

    pred_iter2 = datahelper.get_iter((a, b, c))
    t3 = time.time()
    res2 = model.predict(pred_iter2).asnumpy()
    t4 = time.time()
    print t2 - t1
    print t4 - t3

    # source
'''
