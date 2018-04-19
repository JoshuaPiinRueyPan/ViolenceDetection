import numpy as np

PATH_TO_TRAIN_SET_CATELOG = 'data/train.txt'
PATH_TO_VAL_SET_CATELOG = 'data/val.txt'
PATH_TO_TEST_SET_CATELOG = 'data/test.txt'

'''
    The input will be (BATCH_SIZE, UNROLLED_SIZE, GROUP_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)
    'BATCH_SIZE': How many Videos should be used for a single step.
    'UNROLLED_SIZE':  How many Frames should be extracted from a video.
    'GROUP_SIZE'  Some net need Many Frames (refered as the GROUP_SIZE) to feed into the
		  net for one inference.  For example, the P2D19_1Fc_1LSTM take two frame
		  images as its input.
'''
GROUP_SIZE = 2
IMAGE_SIZE = 224
IMAGE_CHANNELS = 3

'''
    In this project, categories = {NoFight, Fight}
'''
NUMBER_OF_CATEGORIES = 2



#####################
# Advenced Settings #
#####################
FLOAT_TYPE = np.float32
