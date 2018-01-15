import numpy as np
import copy


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

    stride = 8
    recf = 27
    # value range from 0~23
    bbox[:, 0] = bbox[:, 0] * stride
    bbox[:, 1] = bbox[:, 1] * stride
    bbox[:, 2] = (bbox[:, 2] - 1) * stride + recf
    bbox[:, 3] = (bbox[:, 3] - 1) * stride + recf

    return np.array(bbox)


def img2feat(bbox):
    '''

    :param img_bbox: in format of (x1,y1,x2,y2)
    :return: feat_bbox: in format of (x1,y1,x2,y2)
    '''
    img_bbox = copy.deepcopy(bbox)
    s, r = 8., 43.
    img_bbox = np.floor((img_bbox - r / 2) / s)
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


def sample_on_feat(stride_x=2, stride_y=2, stride_w=2, stride_h=2,
                   ideal_w=12, ideal_h=12, feat_w=24, feat_h=24):
    '''

    :param stride_x:
    :param stride_y:
    :param stride_w:
    :param stride_h:
    :param feat_w:
    :param feat_h:
    :return: bbox on feature map, in format of (x1,y1,x2,y2)
    '''
    feat_boxes = list()
    for x in np.arange(0, feat_w - ideal_w / 2., stride_x):
        for y in np.arange(0, feat_h - ideal_h / 2., stride_y):
            max_w = min(ideal_w * 1.5, feat_w - x)
            max_h = min(ideal_h * 1.5, feat_h - y)
            for w in np.arange(ideal_w * 0.5, max_w + 0.1, stride_w):
                for h in np.arange(ideal_h * 0.5, max_h + 0.1, stride_h):
                    feat_boxes.append([0, x, y, x + w - 1, y + h - 1])

    return np.array(feat_boxes)


def train_data(img, region, stride_w=0.2, stride_h=0.2):
    '''

    :param img:
    :param region: in format of (x,y,w,h)
    :param stride_w:
    :param stride_h:
    :return:
    '''
    from scipy.misc import imresize
    import util
    img_H, img_W, c = np.shape(img)
    x, y, w, h = region
    X, Y, W, H = x - w / 2., y - h / 2., 2 * w, 2 * h
    patches = list()
    for scale_w in np.arange(0.5, 1.6, stride_w):
        for scale_h in np.arange(0.5, 1.6, stride_h):
            W_, H_ = W * scale_w, H * scale_h
            X_, Y_ = x + w / 2. - W_ / 2., y + h / 2. - H_ / 2.
            # in case of out of range
            X_, Y_ = max(0, X_), max(0, Y_)
            W_, H_ = min(img_W - X_, W_), min(img_H - Y_, H_)
            patches.append([X_, Y_, W_, H_])

    data_list = list()
    for patch in patches:
        # crop image as train_data
        X, Y, W, H = patch
        img_patch = imresize(img[int(Y):int(Y + H), int(X):int(X + W), :], [227, 227])
        # ISSUE: change HWC to CHW
        img_patch = img_patch.reshape((3, 227, 227))

        # get region
        label_region = np.array([[227. * (x - X) / W, 227. * (y - Y) / H, 227. * w / W, 227. * h / H]])
        # get 50 pos samples
        # feat_boxes = sample()
        # img_boxes = feat2img(feat_boxes)
        label_feat = x1y2x2y22xywh(img2feat(xywh2x1y1x2y2(label_region)))
        feat_bbox, label = get_samples(label_feat)
        # get train_label
        data_list.append([img_patch, feat_bbox, label])

    return data_list


def get_samples(label_feat, pos_number=200, neg_number=200):
    x, y, w, h = label_feat[0, :]
    feat_bboxes = x1y2x2y22xywh(sample_on_feat(1, 1, 1, 1, w, h)[:, 1:5])
    import util
    rat = util.overlap_ratio(label_feat, feat_bboxes)
    pos_samples = feat_bboxes[rat > 0.7, :]
    neg_samples = feat_bboxes[rat < 0.3, :]
    # print 'pos:%d ,neg:%d, all:%d;' % (pos_samples.shape[0], neg_samples.shape[0], feat_bboxes.shape[0])
    # select samples
    # ISSUE: what if pos_samples.shape[0] < pos_number?
    import random
    pos_index = random.sample(range(0, pos_samples.shape[0]), pos_number)
    neg_index = random.sample(range(0, neg_samples.shape[0]), pos_number)

    pos = np.hstack((np.zeros((pos_number, 1)), pos_samples[pos_index, :]))
    neg = np.hstack((np.zeros((neg_number, 1)), neg_samples[neg_index, :]))
    return np.vstack((pos, neg)), \
           np.hstack((np.ones((pos_number,)), np.zeros((neg_number,))))
