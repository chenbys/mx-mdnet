import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np


def show_tracking(img, bboxes):
    bboxes = np.array(bboxes)
    if bboxes.ndim == 1:
        bboxes = np.array([bboxes])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img.astype('uint8'))
    for i in range(bboxes.shape[0]):
        ax.add_patch(patches.Rectangle((bboxes[i, :][0], bboxes[i, :][1]), bboxes[i, :][2], bboxes[i, :][3],
                                       linewidth=2, edgecolor='red', facecolor='none'))
    fig.show()
