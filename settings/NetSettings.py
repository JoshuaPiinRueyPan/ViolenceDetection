#from src.net.old_G1D19_1Fc_1LSTM import *
from src.net.G1D19_1Fc_1LSTM import *
#from src.net.G1D19_1Fc_2LSTM import *
#from src.net.D19_3Fc import *

def GetNetwork(inputImage_, batchSize_, unrolledSize_, isTraining_, trainingStep_):
	print("\n Using Network: ", Net.__module__, "\n")
	return Net(inputImage_, batchSize_, unrolledSize_, isTraining_, trainingStep_)
