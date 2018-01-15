import mxnet as mx
import numpy as np
from symbol import mdnet

image_patch = mx.symbol.Variable(name='image_patch')
feat_bbox = mx.symbol.Variable(name='feat_bbox')
label = mx.symbol.Variable(name='label')
net, smooth_l1, feat = mdnet.get_mdnet(image_patch, feat_bbox, label)
print smooth_l1.infer_shape(image_patch=(1, 3, 227, 227), feat_bbox=(1261, 5), label=(1521, 2))[1]
print net.infer_shape(image_patch=(1, 3, 227, 227), feat_bbox=(1261, 5), label=(1521, 2))[1]
print feat.infer_shape(image_patch=(1, 3, 227, 227))[1]
