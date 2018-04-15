import tensorflow as tf
from src.net.SubnetBase import NetBase
from src.layers.BasicLayers import *
import settings.NetSettings as netSettings
import settings.LayerSettings as layerSettings
import numpy as np

DARKNET19_MODEL_PATH = 'data/darknet19/darknet19.pb'
MSRA_INITIALIZER = tf.contrib.layers.variance_scaling_initializer()

class Net(NetworkBase):
	def __init__(self, inputImage_, BATCH_SIZE_, UNROLLED_SIZE_, isTraining_, trainingStep_):
		self._inputImage = inputImage_
		self._BATCH_SIZE = BATCH_SIZE_
		self._UNROLLED_SIZE = UNROLLED_SIZE_
		self._isTraining = isTraining_
		self._trainingStep = trainingStep_
	
		self._DROPOUT_VALUE = 0.5
		self._NUMBER_OF_NEURONS_IN_LSTM_1 = 1024

	def Build(self):
		darknet19_GraphDef = tf.GraphDef()
		self._inputImage = tf.reshape(self._inputImage, [self._BATCH_SIZE*self._UNROLLED_SIZE, -1])

		with tf.name_scope("DarkNet19"):
			with open(DARKNET19_MODEL_PATH, 'rb') as modelFile:
				darknet19_GraphDef.ParseFromString(modelFile.read())
				listOfOperations = tf.import_graph_def(darknet19_GraphDef,
									input_map={"input": self._inputImage},
									return_elements=["BiasAdd_13"])
#									return_elements=["32-leaky"])
#									return_elements=["BiasAdd_14"])
#									return_elements=["34-leaky"])
#									return_elements=["BiasAdd_15"])
#									return_elements=["36-leaky"])
#									return_elements=["BiasAdd_16"])
#									return_elements=["38-leaky"])
#									return_elements=["BiasAdd_17"])
#									return_elements=["40-leaky"])
#									return_elements=["Pad_18"])
#									return_elements=["41-convolutional"])
#									return_elements=["BiasAdd_18"])
				lastOp = listOfOperations[-1]
				darknetOutput = lastOp.outputs[0]
			
		print("darknetOutput.shape = ", darknetOutput.shape)

		with tf.name_scope("Fc_ConcatPair"):
			out = FullyConnectedLayer('Fc1', out, numberOfOutputs_=2048)
			net, updateVariablesOp1 = BatchNormalization('BN1', net, isConvLayer_=False,
								     isTraining_=self._isTraining, currentStep_=self._trainingStep)
			out = tf.reshape(out, [self._BATCH_SIZE, self._UNROLLED_SIZE, -1])

		print("LSTM input.shape = ", out.shape)

		out, self._stateTensorOfLSTM_1, self._statePlaceHolderOfLSTM_1, self._lstm_1 = LSTM("LSTM_1",
												    out,
												    self._NUMBER_OF_NEURONS_IN_LSTM_1,
												    dropoutProb_=0.5,
												    isTraining_=self._isTraining)
		self._logits = FullyConnectedLayer('Fc3', net, numberOfOutputs_=netSettings.NUMBER_OF_CATEGORIES)
		print("Fc_final logits.shape = ", self.logits.shape)  # The output shape is (batchSize, unrolledSize, NUMBER_OF_CATEGORIES)

		self._updateOp = tf.group(updateVariablesOp1)

	@property
	def logitsOp(self):
		return self._logits

	@property
	def updateOp(self):
		return self._updateOp


	def GetListOfStatesTensorInLSTMs(self):
		'''
		    You should Not Only sess.run() the net.logits, but also this listOfTensors
		    to get the States of LSTM.  And assign it to PlaceHolder next time.
		    ex:
			>> tupleOfResults = sess.run( [out] + net.GetListOfStatesTensorInLSTMs(), ...)
			>> listOfResults = list(tupleOfResults)
			>> output = listOfResults.pop(0)
			>> listOfStates = listOfResults

		    See GetFeedDictOfLSTM() method as well
		'''
		return [self._stateTensorOfLSTM_1]


	def GetFeedDictOfLSTM(self, BATCH_SIZE_, listOfPreviousStateValues_=None):
		'''
		      This function will return a dictionary that contained the PlaceHolder-Value map
		    of the LSTM states.
		      You can use this function as follows:
		    >> feed_dict = { netInput : batchOfImages }
		    >> feedDictOFLSTM = net.GetLSTM_Feed_Dict(BATCH_SIZE, listOfPreviousStateValues)
		    >> tupleOfOutputs = sess.run( [out] + net.GetListOfStatesTensorInLSTMs(),
						  feed_dict = feed_dict.update(feedDictOFLSTM) ) 
		    >> listOfOutputs = list(tupleOfOutputs)
		    >> output = listOfOutputs.pop(0)
		    >> listOfPreviousStateValues = listOfOutputs.pop(0)
		'''
		if listOfPreviousStateValues_ == None:
			'''
			    For the first time (or, the first of Unrolls), there's no previous state,
			    return zeros state.
			'''
			#initialCellState = tuple( [np.zeros([BATCH_SIZE_, self._NUMBER_OF_NEURONS_IN_LSTM_1])] * 2 )
			#initialCellState = tf.nn.rnn_cell.LSTMStateTuple(initialCellState[0], initialCellState[1])
			initialCellState = self._lstm_1.zeros_states(BATCH_SIZE_, layerSettings.FLOAT_TYPE)

			return {self._statePlaceHolderOfLSTM_1 : initialCellState }
		else:
			if len(listOfPreviousStateValues_) != 1:
				errorMessage = "len(listOfPreviousStateValues_) = " + str( len(listOfPreviousStateValues_) )
				errorMessage += "; However, the expected lenght is 1.\n"
				errorMessage += "\t Do you change the Network Structure, such as Add New LSTM?\n"
				errorMessage += "\t Or, do you add more tensor to session.run()?\n"

			return { self._statePlaceHolderOfLSTM_1 : listOfPreviousStateValues_[0] }

