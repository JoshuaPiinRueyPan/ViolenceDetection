import numpy as np

PATH_TO_TRAIN_SET_CATELOG = 'data/train.txt'
PATH_TO_VAL_SET_CATELOG = 'data/val.txt'
PATH_TO_TEST_SET_CATELOG = 'data/test.txt'

'''
    The input will be (BATCH_SIZE, UNROLLED_SIZE, GROUPED_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)
    'BATCH_SIZE': How many Videos should be used for a single step.
    'UNROLLED_SIZE':  How many Frames should be extracted from a video.
    'GROUPED_SIZE'  Some net need Many Frames (refered as the GROUPED_SIZE) to feed into the
		  net for one inference.  For example, the P2D19_1Fc_1LSTM take two frame
		  images as its input.
'''
GROUPED_SIZE = 2
IMAGE_SIZE = 224
IMAGE_CHANNELS = 3

'''
    In this project, categories = {NoFight, Fight}
'''
NUMBER_OF_CATEGORIES = 2


#########################
#  Uncommen Adjustment  #
#########################
NO_FIGHT_LABEL = [1., 0.]
FIGHT_LABEL = [0., 1.]


#####################
# Advenced Settings #
#####################
FLOAT_TYPE = np.float32
'''
    Following control the timeout of LoadData Thread.
    Recommand values:
	BATCH_SIZE=4, No DataAug, TIMEOUT_FOR_WAIT_QUEUE = 10
	BATCH_SIZE=4, DataAug, TIMEOUT_FOR_WAIT_QUEUE = 40
	BATCH_SIZE=20, No DataAug, TIMEOUT_FOR_WAIT_QUEUE = 20
	BATCH_SIZE=40, No DataAug, TIMEOUT_FOR_WAIT_QUEUE = 40
	BATCH_SIZE=40, DataAug, TIMEOUT_FOR_WAIT_QUEUE = 100
'''
TIMEOUT_FOR_WAIT_QUEUE = 100
