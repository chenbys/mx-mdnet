#!/home/chen/anaconda2/bin/python
import vot
import sys
import time
import mxnet as mx
from scipy.misc import imresize

handle = vot.VOT("rectangle")
selection = handle.region()

# Process the first frame
imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

while True:
    imagefile = handle.frame()
    if not imagefile:
        break

    handle.report(selection)
    time.sleep(0.01)
