# -*-coding:utf- 8-*-

from scipy.misc import imresize
import numpy as np
import copy
from setting import constant


def restore_img_bbox(patch_bboxes, restore_info):
    '''
        将patch上的bbox，恢复到原图上的bbox
    :param opt_patch_bbox: x,y,w,h
    :param restore_info: (img_W, img_H, X, Y, W, H)，其中img_WH是原图的size，XYWH是原图上的patch_bbox
    :return: (x,y,w,h) 原图上的bbox
    '''

    px = patch_bboxes[:, 0]
    py = patch_bboxes[:, 1]
    pw = patch_bboxes[:, 2]
    ph = patch_bboxes[:, 3]

    img_W, img_H, X, Y, W, H = restore_info
    px, py = W / 227. * px + X - img_W, H / 227. * py + Y - img_H
    pw, ph = W / 227. * pw, H / 227. * ph
    px = np.max((np.zeros_like(px), px), axis=0)
    py = np.max((np.zeros_like(py), py), axis=0)
    pw = np.min((img_W - px, pw), axis=0)
    ph = np.min((img_H - py, ph), axis=0)

    # x, y = W / 227. * xo + X - img_W, H / 227. * yo + Y - img_H
    # w, h = W / 227. * wo, H / 227. * ho

    # CUT in case out of range
    # x, y = max(0, x), max(0, y)
    # w, h = min(w, img_W - x), min(h, img_H - y)
    img_bboxes = np.vstack((px, py, pw, ph)).transpose()
    return img_bboxes


def crop_img(img, bbox):
    return img[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2]), :]


def feat2img(box):
    '''
        convert bboxes on feat to img
    :param box: bbox on feature map in format of (x1,y1,x2,y2)
    :return: bbox on img in format of (x,y,w,h)
    '''
    bbox = copy.deepcopy(box)
    # in format of (x,y,w,h)
    bbox[:, 2] = bbox[:, 2] - bbox[:, 0] + 1
    bbox[:, 3] = bbox[:, 3] - bbox[:, 1] + 1

    # value range from 0~23
    bbox[:, 0] = bbox[:, 0] * constant.stride
    bbox[:, 1] = bbox[:, 1] * constant.stride
    bbox[:, 2] = (bbox[:, 2] - 1) * constant.stride + constant.recf
    bbox[:, 3] = (bbox[:, 3] - 1) * constant.stride + constant.recf

    return np.array(bbox)


def img2feat(bbox):
    '''

    :param img_bbox: in format of (x1,y1,x2,y2)
    :return: feat_bbox: in format of (x1,y1,x2,y2)
    '''
    img_bbox = copy.deepcopy(bbox)
    img_bbox = np.floor((img_bbox - constant.recf / 2) / constant.stride)
    img_bbox[img_bbox < 0] = 0
    img_bbox[img_bbox > 23] = 23
    return img_bbox


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
    rect1 = np.array(rect1)
    rect2 = np.array(rect2)
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
