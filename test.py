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


def test_overlap():
    import util
    a = np.array([0., 0., 5., 5.])
    b = np.array([1., 1., 5., 5.])
    r = util.overlap_ratio(a, np.array([b, b, a, [5, 5, 5, 5]]))
    print r


def test_feat2img():
    import sample
    feat_bbox = np.array([[0, 0, 4, 4],
                          [0, 0, 23, 23]])
    img_bbox = sample.feat2img(feat_bbox)
    print img_bbox


def test_train_on_one_frame():
    img_path = '/Users/chenjunjie/workspace/OTB/Liquor/img/0001.jpg'
    region = np.array([256, 152, 73, 210])
    import train
    train.train_on_one_frame(img_path, region)


def test_smooth():
    import mxnet as mx
    score = np.array([[0.1, 0.9], [0.3, 0.7], [5, 6]])
    score = mx.ndarray.array(score)
    print score.shape
    label = np.array([[2, 1], [3, 4], [5, 0]])
    label = mx.ndarray.array(label)
    print mx.ndarray.smooth_l1(data=score - label, scalar=1)


def test_repet():
    import matplotlib.pyplot as plt
    import numpy as np
    img = plt.imread('s.jpg')
    print img.shape
    pad_img = np.concatenate((img, img, img), 0)
    pad_img = np.concatenate((pad_img, pad_img, pad_img), 1)
    print pad_img.shape
    plt.imshow(pad_img)
    plt.show()


if __name__ == '__main__':
    test_repet()
