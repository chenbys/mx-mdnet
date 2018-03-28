import numpy as np
import matplotlib.pyplot as plt


def get_res(file_path='/ball1/ball1_001.txt',
            dir_path='/home/chen/vot-toolkit/cmdnet-workspace/results/cmdnet/baseline'):
    seqs = []
    regions = []
    for line in open(dir_path + file_path):
        r = line.replace('\n', '').split(',')
        r = [float(x) for x in r]
        x = r[::2]
        y = r[1::2]
        x1, y1, x2, y2 = min(x), min(y), max(x), max(y)


if __name__ == '__main__':
    get_res()
