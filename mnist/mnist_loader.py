import gzip, struct
import numpy as np


def _read(image, label):
    minist_dir = 'mnist/'
    with gzip.open(minist_dir + label) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(minist_dir + image, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return image, label


def get_data():
    train_img, train_label = _read(
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz')
    test_img, test_label = _read(
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz')
    return [np.reshape(train_img, (60000, 1, 28, 28)).astype('float32'), train_label,
            np.reshape(test_img, (10000, 1, 28, 28)).astype('float32'), test_label]
