PATH_TO_TRAIN_SET_CATELOG = 'data/train.txt'
PATH_TO_VAL_SET_CATELOG = 'data/val.txt'
PATH_TO_TEST_SET_CATELOG = 'data/test.txt'

'''
    The input will be (BATCH_SIZE*UNROLLED_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)
    'BATCH_SIZE': How many Videos should be used for a single step.
    'NUMBER_OF_UNROLLS':  How many Frames should be extracted from a video.
'''
BATCH_SIZE = 12
UNROLLED_SIZE = 2
IMAGE_SIZE = 448
