from mnist import mnist_loader
import mxnet as mx
import numpy as np


class Datahelper(object):
    def __init__(self, dataset_name, batch_size):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        # (6w,1,28,28),(6w,),(1w,1,28,28),(1w,)
        self.train_img, self.train_label, self.test_img, self.test_label = mnist_loader.get_data()
        self.data_desc = mx.io.DataDesc('data', (batch_size, 1, 28, 28), np.float32, 'NCHW')

    def get_train_iter(self, domain_idx, k):
        return mx.io.NDArrayIter(self.train_img / 255.0, self.train_label, self.batch_size,
                                 label_name='branch' + str(domain_idx) + '_softmax_label')

    def get_val_iter(self, domain_idx, k):
        return mx.io.NDArrayIter(self.test_img / 255.0, self.test_label, self.batch_size,
                                 label_name='branch' + str(domain_idx) + '_softmax_label')

    def get_data_desc(self):
        return self.data_desc

    def get_label_desc(self, domain_idx):
        return mx.io.DataDesc('branch' + str(domain_idx) + '_softmax_label',
                              (self.batch_size,), np.float32, 'NCHW')
