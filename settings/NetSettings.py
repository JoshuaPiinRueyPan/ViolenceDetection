#from src.net.D19_3Fc import *

#from src.net.old_G1D19_1Fc_1LSTM import *
#from src.net.G1D19_1Fc_1LSTM import *
#from src.net.G1D19_1Fc_2LSTM import *
#from src.net.G1D19_ResHB_ResHB_1LSTM import *

#from src.net.G2D19_1Fc_1LSTM import *
#from src.net.G2D19_BN_1Fc_1LSTM import *
#from src.net.G2D19_1Fc_2LSTM import *
#from src.net.G2D19_Conv_1LSTM import *
#from src.net.G2D19_BN_Conv_1LSTM import *
#from src.net.G2D19_VGG_1LSTM import *
#from src.net.G2D19_P2OF_CNN_1LSTM import *
#from src.net.G2D19_P2OF_ResBB_1LSTM import *
from src.net.G2D19_P2OF_ResHB_1LSTM import *   # Champion
#from src.net.G2D19_P2OF_ResHB_passthrough_1LSTM import *

def GetNetwork(inputImage_, batchSize_, unrolledSize_, isTraining_, trainingStep_):
	print("\n Using Network: ", Net.__module__, "\n")
	return Net(inputImage_, batchSize_, unrolledSize_, isTraining_, trainingStep_)
