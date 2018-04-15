import numpy as np

PATH_TO_TRAIN_SET_CATELOG = 'data/train.txt'
PATH_TO_VAL_SET_CATELOG = 'data/val.txt'
PATH_TO_TEST_SET_CATELOG = 'data/test.txt'

'''
    The input will be (BATCH_SIZE*UNROLLED_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)
    'BATCH_SIZE': How many Videos should be used for a single step.
    'UNROLLED_SIZE':  How many Frames should be extracted from a video.
'''
IMAGE_SIZE = 448
IMAGE_CHANNELS = 3

#####################
# Advenced Settings #
#####################
DATA_QUEUE_MAX_SIZE = 30
FLOAT_TYPE = np.float32
