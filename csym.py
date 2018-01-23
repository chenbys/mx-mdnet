import mxnet as mx
from kit import p


def get_mdnet(prefix=''):
    '''
    shape of image_patch: (3,227,227)
    :param image_patch: symbol of image data
    :param feat_bbox: symbol of bbox on feat
    :param prefix:
    :return: symbol of mdnet
    '''
    p('use {0,1} cross-entropy loss')
    image_patch = mx.symbol.Variable(name='image_patch')
    feat_bbox = mx.symbol.Variable(name='feat_bbox')
    label = mx.symbol.Variable(name='label')
    label_ = mx.symbol.reshape(label, (-1,), name='label_')
    feat_bbox_ = mx.symbol.reshape(feat_bbox, (-1, 5), name='feat_bbox_')
    # (1,3,211,211)
    conv1 = mx.symbol.Convolution(data=image_patch, kernel=(7, 7), stride=(2, 2), num_filter=96, name=prefix + 'conv1')
    rl1 = mx.symbol.Activation(data=conv1, act_type='relu', name=prefix + 'rl1')
    lrn1 = mx.symbol.LRN(data=rl1, knorm=2, nsize=5, alpha=1e-4, beta=0.75, name=prefix + 'lrn1')
    pool1 = mx.symbol.Pooling(data=lrn1, pool_type='max', kernel=(3, 3), stride=(2, 2), name=prefix + 'pool1')
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5, 5), stride=(2, 2), num_filter=256, name=prefix + 'conv2')
    rl2 = mx.symbol.Activation(data=conv2, act_type='relu', name=prefix + 'rl2')
    lrn2 = mx.symbol.LRN(data=rl2, alpha=1e-4, beta=0.75, knorm=2, nsize=5, name=prefix + 'lrn2')
    # shape of bn2: (1,512,22,22) , recf=27
    conv3 = mx.symbol.Convolution(data=lrn2, kernel=(3, 3), stride=(1, 1), num_filter=512, name=prefix + 'conv3')
    # shape of conv3: (1,512,24,24), recf=43
    rois = mx.symbol.ROIPooling(data=conv3, rois=feat_bbox_, pooled_size=(3, 3), spatial_scale=1.,
                                name=prefix + 'roi_pool')
    fc4 = mx.symbol.Convolution(data=rois, kernel=(3, 3), stride=(1, 1), num_filter=512, name=prefix + 'fc4')
    relu4 = mx.symbol.Activation(data=fc4, act_type='relu', name=prefix + 'relu4')
    drop4 = mx.symbol.Dropout(data=relu4, p=0.5, name=prefix + 'drop4')

    fc5 = mx.symbol.Convolution(data=drop4, kernel=(1, 1), stride=(1, 1), num_filter=512, name=prefix + 'fc5')
    relu5 = mx.symbol.Activation(data=fc5, act_type='relu', name=prefix + 'relu5')
    drop5 = mx.symbol.Dropout(data=relu5, p=0.5, name=prefix + 'drop5')
    score = mx.symbol.Convolution(data=drop5, kernel=(1, 1), stride=(1, 1), num_filter=2, name=prefix + 'score')
    score_ = mx.symbol.Reshape(data=score, shape=(-1, 2), name=prefix + 'score_')

    loss = mx.symbol.SoftmaxOutput(data=score_, label=label_, name=prefix + 'loss')

    # return loss, conv1, lrn2
    return loss


def get_mdnet_with_smooth_l1_loss(prefix=''):
    '''
    shape of image_patch: (3,227,227)
    :param image_patch: symbol of image data
    :param feat_bbox: symbol of bbox on feat
    :param prefix:
    :return: symbol of mdnet
    '''
    p('use smooth_l1 loss')
    image_patch = mx.symbol.Variable(name='image_patch')
    feat_bbox = mx.symbol.Variable(name='feat_bbox')
    # label: (1,K)
    label = mx.symbol.Variable(name='label')
    # label_: (K,)
    label_ = mx.symbol.reshape(label, (-1,), name='label_')
    feat_bbox_ = mx.symbol.reshape(feat_bbox, (-1, 5), name='feat_bbox_')
    # (1,3,211,211)
    conv1 = mx.symbol.Convolution(data=image_patch, kernel=(7, 7), stride=(2, 2), num_filter=96, name=prefix + 'conv1')
    rl1 = mx.symbol.Activation(data=conv1, act_type='relu', name=prefix + 'rl1')
    lrn1 = mx.symbol.LRN(data=rl1, knorm=2, nsize=5, alpha=1e-4, beta=0.75, name=prefix + 'lrn1')
    pool1 = mx.symbol.Pooling(data=lrn1, pool_type='max', kernel=(3, 3), stride=(2, 2), name=prefix + 'pool1')
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5, 5), stride=(2, 2), num_filter=256, name=prefix + 'conv2')
    rl2 = mx.symbol.Activation(data=conv2, act_type='relu', name=prefix + 'rl2')
    lrn2 = mx.symbol.LRN(data=rl2, alpha=1e-4, beta=0.75, knorm=2, nsize=5, name=prefix + 'lrn2')
    # shape of bn2: (1,512,22,22) , recf=27
    conv3 = mx.symbol.Convolution(data=lrn2, kernel=(3, 3), stride=(1, 1), num_filter=512, name=prefix + 'conv3')
    # shape of conv3: (1,512,24,24), recf=43
    rois = mx.symbol.ROIPooling(data=conv3, rois=feat_bbox_, pooled_size=(3, 3), spatial_scale=1.,
                                name=prefix + 'roi_pool')
    fc4 = mx.symbol.Convolution(data=rois, kernel=(3, 3), stride=(1, 1), num_filter=512, name=prefix + 'fc4')
    relu4 = mx.symbol.Activation(data=fc4, act_type='relu', name=prefix + 'relu4')
    drop4 = mx.symbol.Dropout(data=relu4, p=0.5, name=prefix + 'drop4')

    fc5 = mx.symbol.Convolution(data=drop4, kernel=(1, 1), stride=(1, 1), num_filter=512, name=prefix + 'fc5')
    relu5 = mx.symbol.Activation(data=fc5, act_type='relu', name=prefix + 'relu5')
    drop5 = mx.symbol.Dropout(data=relu5, p=0.5, name=prefix + 'drop5')
    score = mx.symbol.Convolution(data=drop5, kernel=(1, 1), stride=(1, 1), num_filter=2, name=prefix + 'score')
    score_ = mx.symbol.Reshape(data=score, shape=(-1, 2), name=prefix + 'score_')
    # score_: (K,2)
    softmax = mx.symbol.softmax(data=score_, name='softmax')
    # pos_pred: (K,)
    pos_pred = mx.symbol.slice(softmax, begin=(None, 0), end=(None, 1)).reshape((-1,), name=prefix + 'pos_pred')
    smooth_l1 = mx.symbol.smooth_l1(data=pos_pred - label_, scalar=1, name=prefix + 'smooth_l1')
    loss = mx.symbol.MakeLoss(data=smooth_l1, normalization='batch')

    # return loss, conv1, lrn2
    return loss


def get_mdnet_c(prefix=''):
    '''
    shape of image_patch: (3,227,227)
    :param image_patch: symbol of image data
    :param feat_bbox: symbol of bbox on feat
    :param prefix:
    :return: symbol of mdnet
    '''
    p('use smooth_l1 loss and cnet')
    image_patch = mx.symbol.Variable(name='image_patch')
    feat_bbox = mx.symbol.Variable(name='feat_bbox')
    # label: (1,K)
    label = mx.symbol.Variable(name='label')
    # label_: (K,)
    label_ = mx.symbol.reshape(label, (-1,), name='label_')
    feat_bbox_ = mx.symbol.reshape(feat_bbox, (-1, 5), name='feat_bbox_')
    # (1,3,211,211)
    conv1 = mx.symbol.Convolution(data=image_patch, kernel=(7, 7), stride=(2, 2), num_filter=96, name=prefix + 'conv1')
    bn1 = mx.symbol.BatchNorm(conv1)
    lrn1 = mx.symbol.LRN(data=bn1, knorm=2, nsize=5, alpha=1e-4, beta=0.75, name=prefix + 'lrn1')
    pool1 = mx.symbol.Pooling(data=lrn1, pool_type='max', kernel=(3, 3), stride=(2, 2), name=prefix + 'pool1')
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5, 5), stride=(2, 2), num_filter=256, name=prefix + 'conv2')
    bn2 = mx.symbol.BatchNorm(conv2)
    lrn2 = mx.symbol.LRN(data=bn2, alpha=1e-4, beta=0.75, knorm=2, nsize=5, name=prefix + 'lrn2')
    # shape of bn2: (1,512,22,22) , recf=27
    conv3 = mx.symbol.Convolution(data=lrn2, kernel=(3, 3), stride=(1, 1), num_filter=512, name=prefix + 'conv3')
    # shape of conv3: (1,512,24,24), recf=43
    rois = mx.symbol.ROIPooling(data=conv3, rois=feat_bbox_, pooled_size=(3, 3), spatial_scale=1.,
                                name=prefix + 'roi_pool')
    fc4 = mx.symbol.Convolution(data=rois, kernel=(3, 3), stride=(1, 1), num_filter=512, name=prefix + 'fc4')
    relu4 = mx.symbol.Activation(data=fc4, act_type='relu', name=prefix + 'relu4')
    drop4 = mx.symbol.Dropout(data=relu4, p=0.5, name=prefix + 'drop4')

    fc5 = mx.symbol.Convolution(data=drop4, kernel=(1, 1), stride=(1, 1), num_filter=512, name=prefix + 'fc5')
    relu5 = mx.symbol.Activation(data=fc5, act_type='relu', name=prefix + 'relu5')
    drop5 = mx.symbol.Dropout(data=relu5, p=0.5, name=prefix + 'drop5')
    score = mx.symbol.Convolution(data=drop5, kernel=(1, 1), stride=(1, 1), num_filter=2, name=prefix + 'score')
    score_ = mx.symbol.Reshape(data=score, shape=(-1, 2), name=prefix + 'score_')
    # score_: (K,2)
    softmax = mx.symbol.softmax(data=score_, name='softmax')
    # pos_pred: (K,)
    pos_pred = mx.symbol.slice(softmax, begin=(None, 0), end=(None, 1)).reshape((-1,), name=prefix + 'pos_pred')
    smooth_l1 = mx.symbol.smooth_l1(data=pos_pred - label_, scalar=1, name=prefix + 'smooth_l1')
    loss = mx.symbol.MakeLoss(data=smooth_l1, normalization='batch')

    # return loss, conv1, lrn2
    return loss
