from src.net.P1D19_1Fc_1LSTM import *

def GetNetwork(inputImage_, batchSize_, unrolledSize_, isTraining_, trainingStep_):
	return Net(inputImage_, batchSize_, unrolledSize_, isTraining_, trainingStep_)
