# -*-coding:utf- 8-*-

from scipy.misc import imresize
import numpy as np
import copy
from setting import const


def replace_wh(xybbox, whbbox):
    '''
        将whbbox改变成：中心是xybbox的中心，长宽不变
    :param xybbox:
    :param whbbox:
    :return:
    '''
    x, y, w, h = xybbox
    whbbox = np.array(whbbox)
    whbbox[:, 0] = x + w / 2. - whbbox[:, 2] / 2.
    whbbox[:, 1] = y + h / 2. - whbbox[:, 3] / 2.
    return whbbox.tolist()


def bbox_contain(B, b):
    X, Y, W, H = B
    x, y, w, h = b
    return (X <= x) & (Y <= y) & ((X + W) >= (x + w)) & ((Y + H) >= (y + h))


def central_bbox(region, dx, dy, w_f, h_f, img_W, img_H):
    x, y, w, h = region
    W = min(img_W, w_f * w)
    H = min(img_H, h_f * h)
    X = min(img_W - W, max(0, x + w / 2. - W / 2. + dx * w))
    Y = min(img_H - H, max(0, y + h / 2. - H / 2. + dy * h))

    return [X, Y, W, H]


def get_img_patch(img, region):
    '''
        传入原图和region，
        将一个不超过原图的
            [const.pred_patch_W,H]的
            以region为中心的bbox
            的部分缩放，
            使得region为[107,107]

        X,Y,W,H是原图上的bbox，用来resize到patch_W,patch_H大小的

    :param img: 原图
    :param region: gt
    :return: img_patch.shape = patch_H,patch_W
    '''
    img_H, img_W, c = np.shape(img)
    x, y, w, h = region
    W, H = int(const.pred_patch_W / 107. * w), int(const.pred_patch_H / 107. * h)
    W, H = min(img_W, W), min(img_H, H)
    patch_W, patch_H = int(W * 107. / w), int(H * 107. / h)
    X, Y = int(max(0, min(img_W - W, x + w / 2. - W / 2.))), int(max(0, min(img_H - H, y + h / 2. - H / 2.)))
    img_patch = img[Y:Y + H, X:X + W, :]
    img_patch_ = imresize(img_patch, [patch_H, patch_W])
    return img_patch_, [X, Y, W, H, patch_W, patch_H]


def transform_bbox(gt, restore_info):
    x, y, w, h = gt
    X, Y, W, H, patch_W, patch_H = restore_info
    w_, h_ = w * patch_W / W, h * patch_H / H
    x_, y_ = (x - X) * patch_W / W, (y - Y) * patch_H / H
    return x_, y_, w_, h_


def restore_bboxes(patch_bboxes, restore_info):
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

    X, Y, W, H, patch_W, patch_H = restore_info
    w, h = w_ / patch_W * W, h_ / patch_H * H
    x, y = x_ / patch_W * W, y_ / patch_H * H
    x, y = x + X, y + Y
    img_bboxes = np.vstack((x, y, w, h)).transpose()
    return img_bboxes


def crop_img(img, bbox):
    return img[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2]), :]


def feat2img(box):
    '''
        convert bboxes on feat to img
    :param box: bbox on feature map in format of (x1,y1,x2,y2)
    :return: bbox on img in format of (x,y,w,h)
    '''
    bbox = np.array(copy.deepcopy(box))
    # in format of (x,y,w,h)
    bbox[:, 2] = bbox[:, 2] - bbox[:, 0] + 1
    bbox[:, 3] = bbox[:, 3] - bbox[:, 1] + 1

    # value range from 0~23
    bbox[:, 0] = bbox[:, 0] * const.stride
    bbox[:, 1] = bbox[:, 1] * const.stride
    bbox[:, 2] = (bbox[:, 2] - 1) * const.stride + const.recf
    bbox[:, 3] = (bbox[:, 3] - 1) * const.stride + const.recf

    return np.array(bbox)


def img2feat(bbox):
    '''
        算x1,y1是算感受野的左侧，算x2,y2是算感受野的右侧
        省略了一种情况：img_bbox的w和h小与43
    :param img_bbox: in format of N*(x1,y1,x2,y2)
    :return: feat_bbox: in format of N*(x1,y1,x2,y2)
    '''
    bbox = np.array(copy.deepcopy(bbox))
    bbox[:, 0] = np.round(bbox[:, 0] / const.stride)
    bbox[:, 1] = np.round(bbox[:, 1] / const.stride)
    bbox[:, 2] = np.round((bbox[:, 2] - const.recf) / const.stride)
    bbox[:, 3] = np.round((bbox[:, 3] - const.recf) / const.stride)

    assert np.min(bbox[:, 0] <= bbox[:, 2]), 'W of img_bbox must greater than 43'
    assert np.min(bbox[:, 1] <= bbox[:, 3]), 'H of img_bbox must greater than 43'

    return bbox


def xywh2x1y1x2y2(bbox):
    '''

    :param bbox: (x,y,w,h)
    :return: (x1,y1,x2,y2)
    '''
    bbox = copy.deepcopy(bbox)
    bbox[:, 2] = bbox[:, 0] + bbox[:, 2] - 1
    bbox[:, 3] = bbox[:, 1] + bbox[:, 3] - 1
    return bbox


def x1y2x2y22xywh(bbox):
    '''

    :param bbox: (x1,y1,x2,y2)
    :return: (x,y,w,h)
    '''
    bbox = copy.deepcopy(bbox)
    bbox[:, 2] = bbox[:, 2] - bbox[:, 0] + 1
    bbox[:, 3] = bbox[:, 3] - bbox[:, 1] + 1
    return bbox


def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''
    rect1 = np.array(rect1, dtype='float32')
    rect2 = np.array(rect2, dtype='float32')
    if rect1.ndim == 1:
        rect1 = rect1[None, :]
    if rect2.ndim == 1:
        rect2 = rect2[None, :]

    left = np.maximum(rect1[:, 0], rect2[:, 0])
    right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
    top = np.maximum(rect1[:, 1], rect2[:, 1])
    bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou


def crop_image(img, bbox, img_size=107, padding=16, valid=False):
    x, y, w, h = np.array(bbox, dtype='float32')

    half_w, half_h = w / 2, h / 2
    center_x, center_y = x + half_w, y + half_h

    if padding > 0:
        pad_w = padding * w / img_size
        pad_h = padding * h / img_size
        half_w += pad_w
        half_h += pad_h

    img_h, img_w, _ = img.shape
    min_x = int(center_x - half_w + 0.5)
    min_y = int(center_y - half_h + 0.5)
    max_x = int(center_x + half_w + 0.5)
    max_y = int(center_y + half_h + 0.5)

    if valid:
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(img_w, max_x)
        max_y = min(img_h, max_y)

    if min_x >= 0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        cropped = img[min_y:max_y, min_x:max_x, :]

    else:
        min_x_val = max(0, min_x)
        min_y_val = max(0, min_y)
        max_x_val = min(img_w, max_x)
        max_y_val = min(img_h, max_y)

        cropped = 128 * np.ones((max_y - min_y, max_x - min_x, 3), dtype='uint8')
        cropped[min_y_val - min_y:max_y_val - min_y, min_x_val - min_x:max_x_val - min_x, :] \
            = img[min_y_val:max_y_val, min_x_val:max_x_val, :]

    scaled = imresize(cropped, (img_size, img_size))
    return scaled
