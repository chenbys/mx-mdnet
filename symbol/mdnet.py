import mxnet as mx


# mx.viz.plot_network(branch1, title='plots/branch1', save_format='jpg').view()
# mx.viz.plot_network(branch2, title='plots/branch2', save_format='jpg').view()
# rec of large_feat is 39
# 223->24, stride=8, err=223/2/8=13
# bbox(x1,y1,x2,y2) on large_feat -> bbox(x1,y1,x2,y2) on img

def get_mdnet(image_patch, feat_bbox, label, prefix=''):
    '''
    shape of image_patch: (3,211,211)
    :param image_patch: symbol of image data
    :param feat_bbox: symbol of bbox on feat
    :param prefix:
    :return: symbol of mdnet
    '''
    label_ = mx.symbol.reshape(label, (-1,), name='label_')
    feat_bbox_ = mx.symbol.reshape(feat_bbox, (-1, 5), name='feat_bbox_')
    # (1,3,211,211)
    conv1 = mx.symbol.Convolution(data=image_patch, kernel=(7, 7), stride=(2, 2), num_filter=96, name=prefix + 'conv1')
    rl1 = mx.symbol.Activation(data=conv1, act_type='relu', name=prefix + 'rl1')
    bn1 = mx.symbol.BatchNorm(data=rl1, name=prefix + 'bn1')
    pool1 = mx.symbol.Pooling(data=bn1, pool_type='max', kernel=(3, 3), stride=(2, 2), name=prefix + 'pool1')
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5, 5), stride=(2, 2), num_filter=256, name=prefix + 'conv2')
    rl2 = mx.symbol.Activation(data=conv2, act_type='relu', name=prefix + 'rl2')
    bn2 = mx.symbol.BatchNorm(data=rl2, name=prefix + 'bn2')
    # shape of bn2: (1,512,22,22) , recf=27
    conv3 = mx.symbol.Convolution(data=bn2, kernel=(3, 3), stride=(1, 1), num_filter=512, name=prefix + 'conv3')
    # shape of conv3: (1,512,24,24), recf=43
    rois = mx.symbol.ROIPooling(data=conv3, rois=feat_bbox_, pooled_size=(3, 3), spatial_scale=1.,
                                name=prefix + 'roi_pool')
    # shape of rois: (1521,512,3,3)
    conv4 = mx.symbol.Convolution(data=rois, kernel=(3, 3), stride=(1, 1), num_filter=512, name=prefix + 'conv4')
    relu4 = mx.symbol.Activation(data=conv4, act_type='relu', name=prefix + 'relu4')
    drop4 = mx.symbol.Dropout(data=relu4, p=0.5, name=prefix + 'drop4')

    fc5 = mx.symbol.Convolution(data=drop4, kernel=(1, 1), stride=(1, 1), num_filter=512, name=prefix + 'fc5')
    relu5 = mx.symbol.Activation(data=fc5, act_type='relu', name=prefix + 'relu5')
    drop5 = mx.symbol.Dropout(data=relu5, p=0.5, name=prefix + 'drop5')
    score_ = mx.symbol.Convolution(data=drop5, kernel=(1, 1), stride=(1, 1), num_filter=2, name=prefix + 'score_')
    score = mx.symbol.Reshape(data=score_, shape=(-1, 2), name=prefix + 'score')

    # data: shape(K,nclass)
    # label: shape(K,)
    loss = mx.symbol.SoftmaxOutput(data=score, label=label_, name=prefix + 'loss')

    return loss
