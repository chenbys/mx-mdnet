import mxnet as mx
import cv2
import numpy as np
import sample

def tracking(img_path_list, region):
    # Mdnet init
    param_path='saved/mdnet_otb-vot15.mat'
    # Get first frame and region
    img_path = img_path_list[0]
    img = cv2.imread(img_path)

    # Train mdnet on first frame
    train_data=sample.train_data(img,region)
    mod=train

    # Train bbox reg on first frame

    # MAIN LOOP

    # Get current frame

    # generate candidates

    # forward for score

    # get result

    # prepare training data

    # online training

    # END LOOP

    pass
