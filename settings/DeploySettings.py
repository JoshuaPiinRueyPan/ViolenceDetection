import numpy as np

FLOAT_TYPE = np.float32

INPUT_SIZE = 224
INPUT_CHANNELS = 3

PATH_TO_MODEL_CHECKPOINTS = "data/bestResult/save_epoch_6/ViolenceNet.ckpt"

FIGHT_LABEL = np.array([0.0, 1.0])

'''
      To smooth the Judgement of Fight/NoneFight, count of neighbor frames
    (that has the value) should >= CHANGE_JUDGEMENT_THRESHOLD to change the
    judgement.
      Usually, this value is setted to the same as the frames Threshold that
    produce the Best Accuracy in Validation set.

    ex:
	NetOutput	Judgement
	  False		  False
	  True		  False
	  True		  False
	  True		  True
	  False		  True
	  Flase		  True
	  Flase		  False
'''
CHANGE_JUDGEMENT_THRESHOLD = 3

DISPLAY_IMAGE_SIZE = 500

BORDER_SIZE = 100
FIGHT_BORDER_COLOR = (0, 0, 255)
NO_FIGHT_BORDER_COLOR = (0, 255, 0)

from src.net.P1D19_1Fc_1LSTM import *

def GetNetwork(inputImage_, batchSize_, unrolledSize_, isTraining_, trainingStep_):
	return Net(inputImage_, batchSize_, unrolledSize_, isTraining_, trainingStep_)
