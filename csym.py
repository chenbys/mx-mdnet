import mxnet as mx


def get_mdnet(prefix=''):
    '''
    shape of image_patch: (3,219,219)
    :param image_patch: symbol of image data
    :param feat_bbox: symbol of bbox on feat
    :param prefix:
    :return: symbol of mdnet
    '''
    # p('use {0,1} cross-entropy loss')
    image_patch = mx.symbol.Variable(name='image_patch')
    feat_bbox = mx.symbol.Variable(name='feat_bbox')
    label = mx.symbol.Variable(name='label')
    label_ = mx.symbol.reshape(label, (-1,), name='label_re')
    feat_bbox_ = mx.symbol.reshape(feat_bbox, (-1, 5), name='feat_bbox_re')
    # (1,3,211,211)
    conv1 = mx.symbol.Convolution(data=image_patch, kernel=(7, 7), stride=(2, 2), num_filter=96, name=prefix + 'conv1')
    rl1 = mx.symbol.Activation(data=conv1, act_type='relu', name=prefix + 'rl1')
    lrn1 = mx.symbol.LRN(data=rl1, knorm=2, nsize=5, alpha=1e-4, beta=0.75, name=prefix + 'lrn1')
    pool1 = mx.symbol.Pooling(data=lrn1, pool_type='max', kernel=(3, 3), stride=(2, 2), name=prefix + 'pool1')
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5, 5), stride=(2, 2), num_filter=256, name=prefix + 'conv2')
    rl2 = mx.symbol.Activation(data=conv2, act_type='relu', name=prefix + 'rl2')
    lrn2 = mx.symbol.LRN(data=rl2, alpha=1e-4, beta=0.75, knorm=2, nsize=5, name=prefix + 'lrn2')
    # ideal output shape of lrn2: (11,11) , recf=27, s=8

    conv3 = mx.symbol.Convolution(data=lrn2, kernel=(3, 3), stride=(1, 1), num_filter=512, name=prefix + 'conv3')
    relu3 = mx.symbol.Activation(data=conv3, act_type='relu', name=prefix + 'relu3')
    # ideal output shape of relu3: (9,9) , recf=43, s=8
    # output shape of relu3: (23,23), recf=43, s=8

    rois = mx.symbol.ROIPooling(data=relu3, rois=feat_bbox_, pooled_size=(3, 3), spatial_scale=1.,
                                name=prefix + 'roi_pool')

    fc4 = mx.symbol.Convolution(data=rois, kernel=(3, 3), stride=(1, 1), num_filter=512, name=prefix + 'fc4')
    relu4 = mx.symbol.Activation(data=fc4, act_type='relu', name=prefix + 'relu4')
    drop4 = mx.symbol.Dropout(data=relu4, p=0.7, name=prefix + 'drop4')

    fc5 = mx.symbol.Convolution(data=drop4, kernel=(1, 1), stride=(1, 1), num_filter=512, name=prefix + 'fc5')
    relu5 = mx.symbol.Activation(data=fc5, act_type='relu', name=prefix + 'relu5')
    drop5 = mx.symbol.Dropout(data=relu5, p=0.7, name=prefix + 'drop5')
    score = mx.symbol.Convolution(data=drop5, kernel=(1, 1), stride=(1, 1), num_filter=2, name=prefix + 'score')
    score_ = mx.symbol.Reshape(data=score, shape=(-1, 2), name=prefix + 'score_re')

    loss = mx.symbol.SoftmaxOutput(data=score_, label=label_, name=prefix + 'loss', normalization='batch')

    # return loss, conv1, lrn2
    return loss

# def get_mdnet2(prefix=''):
#     '''
#     shape of image_patch: (3,195,195), idea obj size:(3,107,107)
#     :param image_patch: symbol of image data
#     :param feat_bbox: symbol of bbox on feat
#     :param prefix:
#     :return: symbol of mdnet
#     '''
#     # p('use {0,1} cross-entropy loss')
#     image_patch = mx.symbol.Variable(name='image_patch')
#     feat_bbox = mx.symbol.Variable(name='feat_bbox')
#     label = mx.symbol.Variable(name='label')
#     label_ = mx.symbol.reshape(label, (-1,), name='label_re')
#     feat_bbox_ = mx.symbol.reshape(feat_bbox, (-1, 5), name='feat_bbox_re')
#     # (1,3,211,211)
#     conv1 = mx.symbol.Convolution(data=image_patch, kernel=(7, 7), stride=(2, 2), num_filter=96, name=prefix + 'conv1')
#     relu1 = mx.symbol.Activation(data=conv1, act_type='relu', name=prefix + 'relu1')
#     lrn1 = mx.symbol.LRN(data=relu1, knorm=2, nsize=5, alpha=1e-4, beta=0.75, name=prefix + 'lrn1')
#     pool1 = mx.symbol.Pooling(data=lrn1, pool_type='max', kernel=(3, 3), stride=(2, 2), name=prefix + 'pool1')
#     conv2 = mx.symbol.Convolution(data=pool1, kernel=(5, 5), stride=(2, 2), num_filter=256, name=prefix + 'conv2')
#     relu2 = mx.symbol.Activation(data=conv2, act_type='relu', name=prefix + 'relu2')
#     lrn2 = mx.symbol.LRN(data=relu2, alpha=1e-4, beta=0.75, knorm=2, nsize=5, name=prefix + 'lrn2')
#     # output size of lrn2:(22,22,256),ref of lrn2=27
#     rois = mx.symbol.ROIPooling(data=lrn2, rois=feat_bbox_, pooled_size=(5, 5), spatial_scale=1.,
#                                 name=prefix + 'roi_pool')
#     # output size of lrn2:(5,5,256)
#     conv3 = mx.symbol.Convolution(data=rois, kernel=(3, 3), stride=(1, 1), num_filter=512, name=prefix + 'conv3')
#     # output size of conv3:(3,3,512)
#     relu3 = mx.symbol.Activation(data=conv3, act_type='relu', name=prefix + 'relu3')
#
#     fc4 = mx.symbol.Convolution(data=relu3, kernel=(3, 3), stride=(1, 1), num_filter=512, name=prefix + 'fc4')
#     relu4 = mx.symbol.Activation(data=fc4, act_type='relu', name=prefix + 'relu4')
#     drop4 = mx.symbol.Dropout(data=relu4, p=0.5, name=prefix + 'drop4')
#
#     fc5 = mx.symbol.Convolution(data=drop4, kernel=(1, 1), stride=(1, 1), num_filter=512, name=prefix + 'fc5')
#     relu5 = mx.symbol.Activation(data=fc5, act_type='relu', name=prefix + 'relu5')
#     drop5 = mx.symbol.Dropout(data=relu5, p=0.5, name=prefix + 'drop5')
#     score = mx.symbol.Convolution(data=drop5, kernel=(1, 1), stride=(1, 1), num_filter=2, name=prefix + 'score')
#     score_ = mx.symbol.Reshape(data=score, shape=(-1, 2), name=prefix + 'score_re')
#
#     loss = mx.symbol.SoftmaxOutput(data=score_, label=label_, name=prefix + 'loss')
#
#     # return loss, conv1, lrn2
#     return loss


# def get_mdnet_with_smooth_l1_loss(prefix=''):
#     '''
#     shape of image_patch: (3,227,227)
#     :param image_patch: symbol of image data
#     :param feat_bbox: symbol of bbox on feat
#     :param prefix:
#     :return: symbol of mdnet
#     '''
#     # p('use smooth_l1 loss')
#     image_patch = mx.symbol.Variable(name='image_patch')
#     feat_bbox = mx.symbol.Variable(name='feat_bbox')
#     # label: (1,K)
#     label = mx.symbol.Variable(name='label')
#     # label_: (K,)
#     label_ = mx.symbol.reshape(label, (-1,), name='label_re')
#     feat_bbox_ = mx.symbol.reshape(feat_bbox, (-1, 5), name='feat_bbox_re')
#     # (1,3,211,211)
#     conv1 = mx.symbol.Convolution(data=image_patch, kernel=(7, 7), stride=(2, 2), num_filter=96, name=prefix + 'conv1')
#     rl1 = mx.symbol.Activation(data=conv1, act_type='relu', name=prefix + 'rl1')
#     lrn1 = mx.symbol.LRN(data=rl1, knorm=2, nsize=5, alpha=1e-4, beta=0.75, name=prefix + 'lrn1')
#     pool1 = mx.symbol.Pooling(data=lrn1, pool_type='max', kernel=(3, 3), stride=(2, 2), name=prefix + 'pool1')
#     conv2 = mx.symbol.Convolution(data=pool1, kernel=(5, 5), stride=(2, 2), num_filter=256, name=prefix + 'conv2')
#     rl2 = mx.symbol.Activation(data=conv2, act_type='relu', name=prefix + 'rl2')
#     lrn2 = mx.symbol.LRN(data=rl2, alpha=1e-4, beta=0.75, knorm=2, nsize=5, name=prefix + 'lrn2')
#     # shape of bn2: (1,512,22,22) , recf=27
#     conv3 = mx.symbol.Convolution(data=lrn2, kernel=(3, 3), stride=(1, 1), num_filter=512, name=prefix + 'conv3')
#     # shape of conv3: (1,512,24,24), recf=43
#     rois = mx.symbol.ROIPooling(data=conv3, rois=feat_bbox_, pooled_size=(3, 3), spatial_scale=1.,
#                                 name=prefix + 'roi_pool')
#     fc4 = mx.symbol.Convolution(data=rois, kernel=(3, 3), stride=(1, 1), num_filter=512, name=prefix + 'fc4')
#     relu4 = mx.symbol.Activation(data=fc4, act_type='relu', name=prefix + 'relu4')
#     drop4 = mx.symbol.Dropout(data=relu4, p=0.5, name=prefix + 'drop4')
#
#     fc5 = mx.symbol.Convolution(data=drop4, kernel=(1, 1), stride=(1, 1), num_filter=512, name=prefix + 'fc5')
#     relu5 = mx.symbol.Activation(data=fc5, act_type='relu', name=prefix + 'relu5')
#     drop5 = mx.symbol.Dropout(data=relu5, p=0.5, name=prefix + 'drop5')
#     score = mx.symbol.Convolution(data=drop5, kernel=(1, 1), stride=(1, 1), num_filter=2, name=prefix + 'score')
#     score_ = mx.symbol.Reshape(data=score, shape=(-1, 2), name=prefix + 'score_re')
#     # score_: (K,2)
#     softmax = mx.symbol.softmax(data=score_, name='softmax')
#     # pos_pred: (K,)
#     pos_pred = mx.symbol.slice(softmax, begin=(None, 0), end=(None, 1), name='sliced_pos_pred').reshape((-1,),
#                                                                                                         name=prefix + 'pos_pred')
#     smooth_l1 = mx.symbol.smooth_l1(data=pos_pred - label_, scalar=1, name=prefix + 'smooth_l1')
#     loss = mx.symbol.MakeLoss(data=smooth_l1, normalization='null', grad_scale=1, name='loss')
#
#     # return loss, conv1, lrn2
#     # return loss


# def get_mdnet_with_CE_loss(prefix=''):
#     '''
#         fake CE loss, but the gradient of pos_score is (pos_score - label)
#     shape of image_patch: (3,227,227)
#     :param image_patch: symbol of image data
#     :param feat_bbox: symbol of bbox on feat
#     :param prefix:
#     :return: symbol of mdnet
#     '''
#     # p('use smooth_l1 loss')
#     image_patch = mx.symbol.Variable(name='image_patch')
#     feat_bbox = mx.symbol.Variable(name='feat_bbox')
#     # label: (1,K)
#     label = mx.symbol.Variable(name='label')
#     # label_: (K,)
#     label_ = mx.symbol.reshape(label, (-1,), name='label_re')
#     feat_bbox_ = mx.symbol.reshape(feat_bbox, (-1, 5), name='feat_bbox_re')
#     # (1,3,211,211)
#     conv1 = mx.symbol.Convolution(data=image_patch, kernel=(7, 7), stride=(2, 2), num_filter=96, name=prefix + 'conv1')
#     rl1 = mx.symbol.Activation(data=conv1, act_type='relu', name=prefix + 'rl1')
#     lrn1 = mx.symbol.LRN(data=rl1, knorm=2, nsize=5, alpha=1e-4, beta=0.75, name=prefix + 'lrn1')
#     pool1 = mx.symbol.Pooling(data=lrn1, pool_type='max', kernel=(3, 3), stride=(2, 2), name=prefix + 'pool1')
#     conv2 = mx.symbol.Convolution(data=pool1, kernel=(5, 5), stride=(2, 2), num_filter=256, name=prefix + 'conv2')
#     rl2 = mx.symbol.Activation(data=conv2, act_type='relu', name=prefix + 'rl2')
#     lrn2 = mx.symbol.LRN(data=rl2, alpha=1e-4, beta=0.75, knorm=2, nsize=5, name=prefix + 'lrn2')
#     # shape of bn2: (1,512,22,22) , recf=27
#     conv3 = mx.symbol.Convolution(data=lrn2, kernel=(3, 3), stride=(1, 1), num_filter=512, name=prefix + 'conv3')
#     # shape of conv3: (1,512,24,24), recf=43
#     rois = mx.symbol.ROIPooling(data=conv3, rois=feat_bbox_, pooled_size=(3, 3), spatial_scale=1.,
#                                 name=prefix + 'roi_pool')
#     fc4 = mx.symbol.Convolution(data=rois, kernel=(3, 3), stride=(1, 1), num_filter=512, name=prefix + 'fc4')
#     relu4 = mx.symbol.Activation(data=fc4, act_type='relu', name=prefix + 'relu4')
#     drop4 = mx.symbol.Dropout(data=relu4, p=0.5, name=prefix + 'drop4')
#
#     fc5 = mx.symbol.Convolution(data=drop4, kernel=(1, 1), stride=(1, 1), num_filter=512, name=prefix + 'fc5')
#     relu5 = mx.symbol.Activation(data=fc5, act_type='relu', name=prefix + 'relu5')
#     drop5 = mx.symbol.Dropout(data=relu5, p=0.5, name=prefix + 'drop5')
#     score = mx.symbol.Convolution(data=drop5, kernel=(1, 1), stride=(1, 1), num_filter=2, name=prefix + 'score')
#     score_ = mx.symbol.Reshape(data=score, shape=(-1, 2), name=prefix + 'score_re')
#     # score_: (K,2)
#     # softmax = mx.symbol.softmax(data=score_, name='softmax')
#     # pos_pred: (K,)
#     pos_pred = mx.symbol.slice(score_, begin=(None, 0), end=(None, 1), name='sliced_pos_pred').reshape((-1,),
#                                                                                                        name=prefix + 'pos_pred')
#     smooth_l1 = mx.symbol.smooth_l1(data=pos_pred - label_, scalar=1, name=prefix + 'smooth_l1')
#     loss = mx.symbol.MakeLoss(data=smooth_l1, normalization='null', grad_scale=1, name='loss')
#
#     # return loss, conv1, lrn2
#     # return loss


# def get_mdnet_with_weighted_CE_loss(weight_factor, prefix=''):
#     '''
#         fake CE loss, but the gradient of pos_score is (pos_score - label)
#     shape of image_patch: (3,227,227), rm some weight for just one score output
#     :param image_patch: symbol of image data
#     :param feat_bbox: symbol of bbox on feat
#     :param prefix:
#     :return: symbol of mdnet
#     '''
#     # p('use smooth_l1 loss')
#     image_patch = mx.symbol.Variable(name='image_patch')
#     feat_bbox = mx.symbol.Variable(name='feat_bbox')
#     # label: (1,K)
#     label = mx.symbol.Variable(name='label')
#     # label_: (K,)
#     label_ = mx.symbol.reshape(label, (-1,), name='label_re')
#     feat_bbox_ = mx.symbol.reshape(feat_bbox, (-1, 5), name='feat_bbox_re')
#     # (1,3,211,211)
#     conv1 = mx.symbol.Convolution(data=image_patch, kernel=(7, 7), stride=(2, 2), num_filter=96, name=prefix + 'conv1')
#     rl1 = mx.symbol.Activation(data=conv1, act_type='relu', name=prefix + 'rl1')
#     lrn1 = mx.symbol.LRN(data=rl1, knorm=2, nsize=5, alpha=1e-4, beta=0.75, name=prefix + 'lrn1')
#     pool1 = mx.symbol.Pooling(data=lrn1, pool_type='max', kernel=(3, 3), stride=(2, 2), name=prefix + 'pool1')
#     conv2 = mx.symbol.Convolution(data=pool1, kernel=(5, 5), stride=(2, 2), num_filter=256, name=prefix + 'conv2')
#     rl2 = mx.symbol.Activation(data=conv2, act_type='relu', name=prefix + 'rl2')
#     lrn2 = mx.symbol.LRN(data=rl2, alpha=1e-4, beta=0.75, knorm=2, nsize=5, name=prefix + 'lrn2')
#     # shape of bn2: (1,512,22,22) , recf=27
#     conv3 = mx.symbol.Convolution(data=lrn2, kernel=(3, 3), stride=(1, 1), num_filter=512, name=prefix + 'conv3')
#     # shape of conv3: (1,512,24,24), recf=43
#     rois = mx.symbol.ROIPooling(data=conv3, rois=feat_bbox_, pooled_size=(3, 3), spatial_scale=1.,
#                                 name=prefix + 'roi_pool')
#     fc4 = mx.symbol.Convolution(data=rois, kernel=(3, 3), stride=(1, 1), num_filter=512, name=prefix + 'fc4')
#     relu4 = mx.symbol.Activation(data=fc4, act_type='relu', name=prefix + 'relu4')
#     drop4 = mx.symbol.Dropout(data=relu4, p=0.5, name=prefix + 'drop4')
#
#     fc5 = mx.symbol.Convolution(data=drop4, kernel=(1, 1), stride=(1, 1), num_filter=512, name=prefix + 'fc5')
#     relu5 = mx.symbol.Activation(data=fc5, act_type='relu', name=prefix + 'relu5')
#     drop5 = mx.symbol.Dropout(data=relu5, p=0.5, name=prefix + 'drop5')
#     score = mx.symbol.Convolution(data=drop5, kernel=(1, 1), stride=(1, 1), num_filter=1, name=prefix + 'score')
#     score_ = mx.symbol.Reshape(data=score, shape=(-1), name=prefix + 'score_re')
#     # pos_pred: (K,)
#     smooth_l1 = mx.symbol.smooth_l1(data=score_ - label_, scalar=1, name=prefix + 'smooth_l1')
#     loss = mx.symbol.MakeLoss(data=smooth_l1 * (label_ * weight_factor + 1.), normalization='null',
#                               grad_scale=1, name='loss')
#
#     # return loss, conv1, lrn2
#     return loss
#
#
# def get_pred_sym(prefix=''):
#     '''
#         fake CE loss, but the gradient of pos_score is (pos_score - label)
#     shape of image_patch: (3,227,227), rm some weight for just one score output
#     :param image_patch: symbol of image data
#     :param feat_bbox: symbol of bbox on feat
#     :param prefix:
#     :return: symbol of mdnet
#     '''
#     # p('use smooth_l1 loss')
#     image_patch = mx.symbol.Variable(name='image_patch')
#     feat_bbox = mx.symbol.Variable(name='feat_bbox')
#     # label: (1,K)
#     # label = mx.symbol.Variable(name='label')
#     # label_: (K,)
#     # label_ = mx.symbol.reshape(label, (-1,), name='label_re')
#     feat_bbox_ = mx.symbol.reshape(feat_bbox, (-1, 5), name='feat_bbox_re')
#     # (1,3,211,211)
#     conv1 = mx.symbol.Convolution(data=image_patch, kernel=(7, 7), stride=(2, 2), num_filter=96, name=prefix + 'conv1')
#     rl1 = mx.symbol.Activation(data=conv1, act_type='relu', name=prefix + 'rl1')
#     lrn1 = mx.symbol.LRN(data=rl1, knorm=2, nsize=5, alpha=1e-4, beta=0.75, name=prefix + 'lrn1')
#     pool1 = mx.symbol.Pooling(data=lrn1, pool_type='max', kernel=(3, 3), stride=(2, 2), name=prefix + 'pool1')
#     conv2 = mx.symbol.Convolution(data=pool1, kernel=(5, 5), stride=(2, 2), num_filter=256, name=prefix + 'conv2')
#     rl2 = mx.symbol.Activation(data=conv2, act_type='relu', name=prefix + 'rl2')
#     lrn2 = mx.symbol.LRN(data=rl2, alpha=1e-4, beta=0.75, knorm=2, nsize=5, name=prefix + 'lrn2')
#     # shape of bn2: (1,512,22,22) , recf=27
#     conv3 = mx.symbol.Convolution(data=lrn2, kernel=(3, 3), stride=(1, 1), num_filter=512, name=prefix + 'conv3')
#     # shape of conv3: (1,512,24,24), recf=43
#     rois = mx.symbol.ROIPooling(data=conv3, rois=feat_bbox_, pooled_size=(3, 3), spatial_scale=1.,
#                                 name=prefix + 'roi_pool')
#     fc4 = mx.symbol.Convolution(data=rois, kernel=(3, 3), stride=(1, 1), num_filter=512, name=prefix + 'fc4')
#     relu4 = mx.symbol.Activation(data=fc4, act_type='relu', name=prefix + 'relu4')
#     drop4 = mx.symbol.Dropout(data=relu4, p=0.5, name=prefix + 'drop4')
#
#     fc5 = mx.symbol.Convolution(data=drop4, kernel=(1, 1), stride=(1, 1), num_filter=512, name=prefix + 'fc5')
#     relu5 = mx.symbol.Activation(data=fc5, act_type='relu', name=prefix + 'relu5')
#     drop5 = mx.symbol.Dropout(data=relu5, p=0.5, name=prefix + 'drop5')
#     score = mx.symbol.Convolution(data=drop5, kernel=(1, 1), stride=(1, 1), num_filter=1, name=prefix + 'score')
#     score_ = mx.symbol.Reshape(data=score, shape=(-1), name=prefix + 'score_re')
#     return score_
#     # pos_pred: (K,)
#     # smooth_l1 = mx.symbol.smooth_l1(data=score_ - label_, scalar=1, name=prefix + 'smooth_l1')
#     # loss = mx.symbol.MakeLoss(data=smooth_l1 * (label_ * label_ * weight_factor + 1.), normalization='null',
#     #                           grad_scale=1, name='loss')
#
#     # return loss, conv1, lrn2
#     # return loss
