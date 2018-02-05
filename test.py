import numpy as np


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
    trian_iter = datahelper.get_train_iter(train_data)
    return


if __name__ == '__main__':
    test_train_iter()
