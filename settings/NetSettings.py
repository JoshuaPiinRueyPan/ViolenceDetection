from src.net.P1D19_1Fc_1LSTM import *

'''
    'PAIRED_SIZE': How many Frames of the Conv parts output should be concatenated
		   and send to the RNN parts.  The concatenate of the Conv parts output
		   can also be seem as the calculation of the Optical Flow.
'''
PAIRED_SIZE = 2
'''
    In this project, categories = {NoFight, Fight}
'''
NUMBER_OF_CATEGORIES = 2

def GetNetwork(isTraining_, trainingStep, inputImage_, inputAngle_, groundTruth_):
	return Net(isTraining_, trainingStep, inputImage_, inputAngle_, groundTruth_)
